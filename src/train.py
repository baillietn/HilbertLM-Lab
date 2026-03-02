import os 
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch 
import time
import math
import gc
from torch.utils.data import DataLoader, Subset
from model import NanoLLM
from dataset import TextDataset, SFTDataset
from config import config

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

target_tokens = config['pre_training_target_tokens']
sft_target_tokens = config['stf_target_tokens']

def get_lr(tokens_seen, max_lr):
    
    warmup_tokens = 0.05 * target_tokens      
    annealing_start = 0.675 * target_tokens  
    pretraining_end = target_tokens  
    
    safe_max_lr = max_lr / 2
    min_lr = max_lr * 0.01  

    if tokens_seen < warmup_tokens:
        return max_lr * (tokens_seen / warmup_tokens)

    elif tokens_seen < annealing_start:
        return max_lr
    
    else:
        decay_ratio = (tokens_seen - annealing_start) / (pretraining_end - annealing_start)
        decay_ratio = min(decay_ratio, 1.0) 
        
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (safe_max_lr - min_lr)

def get_stage_info(tokens_seen):
    if tokens_seen < 0.4 * target_tokens: 
        return "STAGE 1"
    
    elif tokens_seen < 0.7 * target_tokens: 
        return "STAGE 2"
    
    elif tokens_seen < 0.9 * target_tokens: 
        return "STAGE 3"
    
    elif tokens_seen < target_tokens: 
        return "STAGE 4"
    
def get_sft_lr(step, total_steps, max_lr):
    warmup_steps = int(total_steps * 0.05)
    min_lr = max_lr * 0.1
    
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def save_checkpoint(model, optimizer, samples_seen, history, last_chinchilla_idx, fp8_recipe, path, quiet=False):
    checkpoint = {
        'samples_seen': samples_seen,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'last_chinchilla_idx': last_chinchilla_idx,
    }
    
    if fp8_recipe is not None:
        try:
            import transformer_engine.pytorch as te
            fp8_state = {}
            
            for name, module in model.named_modules():
                if isinstance(module, (te.Linear, te.LayerNorm)):
                    if hasattr(module, 'fp8_meta') and module.fp8_meta is not None:
                        state_dict = {}
                        if 'scaling_fwd' in module.fp8_meta:
                            scaling_fwd = module.fp8_meta['scaling_fwd']
                            if hasattr(scaling_fwd, 'scale') and scaling_fwd.scale is not None:
                                state_dict['scale_fwd'] = scaling_fwd.scale.cpu().clone()
                            if hasattr(scaling_fwd, 'scale_inv') and scaling_fwd.scale_inv is not None:
                                state_dict['scale_inv_fwd'] = scaling_fwd.scale_inv.cpu().clone()
                            if hasattr(scaling_fwd, 'amax_history') and scaling_fwd.amax_history is not None:
                                state_dict['amax_history_fwd'] = scaling_fwd.amax_history.cpu().clone()
                        
                        if state_dict: 
                            fp8_state[name] = state_dict
            
            if fp8_state:
                checkpoint['fp8_state'] = fp8_state
                if not quiet:
                    print(f"Saving {len(fp8_state)} FP8 modules...")
        except Exception as e:
            print(f"Error while saving FP8 modules: {e}")
    
    torch.save(checkpoint, path)
    if not quiet:
        print(f"Checkpoint saved: {path}")

def is_compiled(model, ckpt):
    state_dict = ckpt['model_state_dict']
        
    model_is_compiled = hasattr(model, '_orig_mod')
    checkpoint_is_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        
    if model_is_compiled and not checkpoint_is_compiled:
        state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
    elif not model_is_compiled and checkpoint_is_compiled:
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    return state_dict

def train(args):
    C = config
    device = C['device']
    max_lr = C['max_lr']
    vocab_size = C['vocab_size']
    block_size = C['block_size']
    d_model = C['d_model']
    n_layer = C['n_layer']
    n_head = C['n_head']
    n_kv_head = C['n_kv_head']
    batch_size = C['batch_size']
    micro_batch_size = args.micro_batch_size or C['micro_batch_size']
    grad_accum_steps = batch_size // micro_batch_size 
    log_interval = C['logging_interval']

    precision = args.precision if args.precision else ("fp8" if C['use_te'] else "bf16")
    use_te = (precision == "fp8")

    if args.sft:
        print("SFT mode was activate")
        
        bin_data_path = os.path.join(C['dataset_path'], "sft_data.bin")
        bin_labels_path = os.path.join(C['dataset_path'], "sft_labels.bin")
        
        if not os.path.exists(bin_data_path) or not os.path.exists(bin_labels_path):
            print("SFT data was not found. Try : python src/get_data.py --prepare-sft")
            return
        
        full_dataset = SFTDataset(bin_data_path, bin_labels_path, block_size)
        
        target_samples = sft_target_tokens // block_size
        
        if len(full_dataset) > target_samples:
            indices = list(range(target_samples))
            train_dataset = Subset(full_dataset, indices)
        else:
            train_dataset = full_dataset
        
        sft_max_lr = 1.5e-4
        checkpoint_path = "checkpoints/sft_checkpoint.pt"
        base_model_path = "checkpoints/final_base_model.pt"
        final_model_path = "checkpoints/nano_llm_chat_final.pt"
        is_sft_mode = True
        
    else:
        train_dataset = TextDataset(C['dataset_path'], block_size=block_size)
        checkpoint_path = C['checkpoint_file_path']
        base_model_path = None
        final_model_path = "checkpoints/final_base_model.pt"
        is_sft_mode = False

    total_tokens_in_file = len(train_dataset.data) if hasattr(train_dataset, 'data') else len(train_dataset) * block_size
    total_samples = len(train_dataset)
    total_steps = total_samples // batch_size

    mode_display = "SFT" if is_sft_mode else "PRETRAIN"
    print(f"Mode {mode_display} : {total_tokens_in_file/1e6:.1f}M tokens -> {total_steps} steps")

    model = NanoLLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        n_layer=n_layer, 
        n_head=n_head, 
        max_len=block_size,
        n_kv_head=n_kv_head,
        use_te=use_te
    )
    
    model.to(device, dtype=torch.bfloat16)
    print(f"Model loaded on {device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    compile_mode = args.compile_mode
    if compile_mode != "none":
        print(f"Compiled CUDA kernels was enabled: {compile_mode}")
        model = torch.compile(model, mode=compile_mode if compile_mode != "default" else None)
    else:
        print("Compiled CUDA kernels was disabled")
    
    print(f"Model was load on {device} with {args.precision.upper()} precision...")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-7, 
        weight_decay=0.1 if not is_sft_mode else 0.01, 
        fused=True
    )

    if args.precision == "fp8":
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
        fp8_recipe = recipe.DelayedScaling(
            margin=0.0, 
            interval=16, 
            fp8_format=recipe.Format.HYBRID,
            amax_history_len=32,
            amax_compute_algo="max",
        )
        print("FP8 Autocast was enabled")
    else:
        fp8_recipe = None
        print("BF16 native was enabled")

    history = {'loss': [], 'lr': [], 'tokens': []}
    samples_seen = 0
    
    if not is_sft_mode:
        chinchilla_interval = 20 * total_params
        last_chinchilla_idx = 0
    else:
        last_chinchilla_idx = 0

    if os.path.exists(checkpoint_path):
        print(f"Restore checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        state_dict = is_compiled(model, ckpt)
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        samples_seen = ckpt.get('samples_seen', ckpt.get('step', 0) * micro_batch_size)
        
        if not is_sft_mode:
            last_chinchilla_idx = ckpt.get('last_chinchilla_idx', 0)

        if 'history' in ckpt:
            history = ckpt['history']
            print(f"History : {len(history['loss'])} points.")
        
        if args.precision == "fp8" and 'fp8_state' in ckpt:
            try:
                import transformer_engine.pytorch as te
                fp8_state = ckpt['fp8_state']
                restored_count = 0
                
                for name, module in model.named_modules():
                    if isinstance(module, (te.Linear, te.LayerNorm)) and name in fp8_state:
                        if hasattr(module, 'fp8_meta') and module.fp8_meta is not None:
                            state = fp8_state[name]
                            if 'scaling_fwd' in module.fp8_meta:
                                scaling_fwd = module.fp8_meta['scaling_fwd']
                                if 'scale_fwd' in state and hasattr(scaling_fwd, 'scale'):
                                    scaling_fwd.scale.copy_(state['scale_fwd'].to(device))
                                if 'scale_inv_fwd' in state and hasattr(scaling_fwd, 'scale_inv'):
                                    scaling_fwd.scale_inv.copy_(state['scale_inv_fwd'].to(device))
                                if 'amax_history_fwd' in state and hasattr(scaling_fwd, 'amax_history'):
                                    scaling_fwd.amax_history.copy_(state['amax_history_fwd'].to(device))
                                restored_count += 1
                
                print(f"FP8 states restored ({restored_count} modules)")
            except Exception as e:
                print(f"Error during FP8 states restorations: {e}")

        tokens_seen_at_resume = samples_seen * block_size
        
        if is_sft_mode:
            global_step_resume = samples_seen // batch_size
            lr = get_sft_lr(global_step_resume, total_steps, sft_max_lr)
            print(f"Resuming SFT from {samples_seen} samples ({tokens_seen_at_resume/1e6:.1f}M tokens)")
        else:
            lr = get_lr(tokens_seen_at_resume, max_lr)
            print(f"Resuming pretrain from {samples_seen} samples ({tokens_seen_at_resume/1e9:.3f}B tokens)")
            print(f"Last Chinchilla checkpoint: #{last_chinchilla_idx}")
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
    
    elif is_sft_mode and os.path.exists(base_model_path):
        print(f"Loading Base Model from {base_model_path}...")
        ckpt = torch.load(base_model_path, map_location='cpu')
        
        state_dict = is_compiled(model, ckpt)
        
        model.load_state_dict(state_dict)
        
        samples_seen = 0
        history = {'loss': [], 'lr': [], 'tokens': []}
        
        del ckpt
        gc.collect()
        print("Base model loaded, optimizer & counters reset for SFT")
    
    elif is_sft_mode:
        print(f"No checkpoint was found. Please ensure one of these exists:")
        print(f"   - {checkpoint_path} (to resume SFT)")
        print(f"   - {base_model_path} (to start SFT from pretrained model)")
        return

    model.train()
    last_log_time = time.time()
    loss_accum = 0.0
    
    start_step = samples_seen // micro_batch_size

    if not is_sft_mode:
        chinchilla_interval = 20 * total_params

    if start_step > 0:
        print(f"Fast-forwarding dataset to step {start_step}...")
        dataset_start_idx = start_step * micro_batch_size
        
        if dataset_start_idx < len(train_dataset):
            indices = range(dataset_start_idx, len(train_dataset))
            train_subset = Subset(train_dataset, indices)
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=micro_batch_size, 
                shuffle=False, 
                num_workers=8, 
                pin_memory=True, 
                persistent_workers=True, 
                prefetch_factor=2
            )
        else:
            print(f"Training seems already finished...")
            return
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=micro_batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=2
        )
    
    interrupted = False
    current_accum_step = 0

    try:
        for step, (x, y) in enumerate(train_loader, start=start_step):

            samples_seen += micro_batch_size
            tokens_seen = samples_seen * block_size
            current_accum_step = step % grad_accum_steps
            
            if not is_sft_mode:
                current_chinchilla_idx = int(tokens_seen // chinchilla_interval)
                if current_chinchilla_idx > last_chinchilla_idx:
                    path = f"checkpoints/chinchilla_{current_chinchilla_idx}.pt"
                    save_checkpoint(model, optimizer, samples_seen, history, current_chinchilla_idx, fp8_recipe, path)
                    last_chinchilla_idx = current_chinchilla_idx

            if step % grad_accum_steps == 0:
                if is_sft_mode:
                    global_step = step // grad_accum_steps
                    lr = get_sft_lr(global_step, total_steps, sft_max_lr)
                else:
                    lr = get_lr(tokens_seen, max_lr)
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            x = x.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()

            if args.precision == "fp8":
                import transformer_engine.pytorch as te
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    logits, loss = model(x, targets=y)
            else:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, loss = model(x, targets=y)

            loss_accum += loss.item()
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step_loss = loss_accum / grad_accum_steps
                loss_accum = 0.0 

                history['loss'].append(step_loss)
                history['lr'].append(lr)
                history['tokens'].append(tokens_seen)

                global_step = (step + 1) // grad_accum_steps

                if global_step % log_interval == 0:
                    dt = time.time() - last_log_time
                    
                    tokens_processed = log_interval * batch_size * block_size
                    tps = tokens_processed / dt
                    
                    if is_sft_mode:
                        progress = (tokens_seen / total_tokens_in_file) * 100
                        avg_display_loss = sum(history['loss'][-log_interval:]) / log_interval
                        print(f"[SFT] Step {samples_seen}/{total_samples} | {tokens_seen/1e6:.1f}M Tok ({progress:.2f}%) | LR: {lr:.2e} | Loss: {avg_display_loss:.4f} | {tps/1000:.1f}k tok/s | {dt:.2f}s")
                    else:
                        stage = get_stage_info(tokens_seen)
                        progress = (tokens_seen / total_tokens_in_file) * 100
                        avg_display_loss = sum(history['loss'][-log_interval:]) / log_interval
                        print(f"[{stage}] Step {samples_seen}/{total_samples} | {tokens_seen/1e9:.3f}B Tok ({progress:.3f}%) | LR: {lr:.2e} | Loss: {avg_display_loss:.4f} | {tps/1000:.1f}k tok/s | {dt:.2f}s")
                    
                    last_log_time = time.time()
        
        print(f"\nTraining is finished ! Saving final model...")
        save_checkpoint(model, optimizer, samples_seen, history, last_chinchilla_idx, fp8_recipe, final_model_path)
        print(f"Model was saved : {final_model_path}")

    except KeyboardInterrupt:
        interrupted = True
        print(f"\nInterruption was detected. Finishing current accumulation step {current_accum_step+1}/{grad_accum_steps}...")
        
        remaining_steps = grad_accum_steps - current_accum_step - 1
        if remaining_steps > 0:
            try:
                temp_loader = DataLoader(
                    train_dataset, 
                    batch_size=micro_batch_size, 
                    shuffle=False, 
                    num_workers=0, 
                    pin_memory=False
                )
                temp_iter = iter(temp_loader)
                
                for _ in range(step + 1):
                    next(temp_iter)
                
                for i in range(remaining_steps):
                    x, y = next(temp_iter)
                    samples_seen += micro_batch_size
                    
                    x = x.to(device, non_blocking=True).long()
                    y = y.to(device, non_blocking=True).long()
                    
                    if args.precision == "fp8":
                        import transformer_engine.pytorch as te
                        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                            logits, loss = model(x, targets=y)
                    else:
                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            logits, loss = model(x, targets=y)
                    
                    loss = loss / grad_accum_steps
                    loss.backward()
                    loss_accum += loss.item()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                print(f"Current accumulation was completed ({grad_accum_steps}/{grad_accum_steps})")
            except Exception as e:
                print(f"Current accumulation was not completed (gradients accumulated: {current_accum_step+1}/{grad_accum_steps})\n{e}")
        
        print(f"Exporting to checkpoint...")
        save_checkpoint(model, optimizer, samples_seen, history, last_chinchilla_idx, fp8_recipe, checkpoint_path)
        print(f"Checkpoint was completed ! Exiting training process.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoLLM")
    
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["bf16", "fp8"], 
        default="bf16",
        help="Precision mode: bf16 (PyTorch native) or fp8 (Transformer Engine)"
    )
    
    parser.add_argument(
        "--compile-mode", 
        type=str, 
        choices=["default", "reduce-overhead", "max-autotune", "none"], 
        default="default",
        help="torch.compile mode"
    )
    
    parser.add_argument(
        "--micro-batch-size", 
        type=int, 
        default=None,
        help="Micro batch size (default: from config)"
    )
    
    parser.add_argument(
        "--sft",
        action="store_true",
        help="Enable SFT mode (Supervised Fine-Tuning)"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print(f"Training configuration {'SFT' if args.sft else 'PRETRAIN'}")
    print("="*50)
    print(f"Precision:        {args.precision.upper()}")
    print(f"Compile Mode:     {args.compile_mode}")
    print(f"Micro Batch Size: {args.micro_batch_size or config['micro_batch_size']}")
    print("="*50)
    
    train(args)
