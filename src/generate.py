import torch
import os
import argparse
import sys
from model import NanoLLM
from tokenizer import SimpleTokenizer
from config import config
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

# Need to be adjusted
#------------------------------------------------
max_context = 512
max_new_tokens = 150
max_prompt_len = max_context - max_new_tokens
         
max_thresh = 12.0
min_thresh = 1.0
#------------------------------------------------

C = config
device = C['device']
max_lr = C['max_lr']
block_size = C['block_size']
d_model = C['d_model']
n_layer = C['n_layer']
n_head = C['n_head']
n_kv_head = C['n_kv_head']
batch_size = C['batch_size']
tokenizer_path = C['tokenizer_path'] + "/tokenizer.json"
llm_chat_file_path = "checkpoints/nano_llm_chat_final.pt"

def apply_repetition_penalty(logits, sequences, penalty, window=50):
    for i in range(logits.shape[0]):
        seq_window = sequences[i, -window:].tolist() if sequences.shape[1] > window else sequences[i].tolist()
        previous_tokens = set(seq_window)
        
        for token_id in previous_tokens:
            if logits[i, token_id] < 0:
                logits[i, token_id] *= penalty
            else:
                logits[i, token_id] /= penalty
    
    return logits

def load_loaded_model(precision=None, model_ckpt=llm_chat_file_path):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = SimpleTokenizer(tokenizer_path)
    if precision == "fp8":
        use_te = True
    elif precision == "bf16":
        use_te = False
    else:
        use_te = C['use_te']
    
    precision_display = precision.upper() if precision else f"CONFIG (use_te={use_te})"
    print(f"Building model (precision={precision_display})...")
    model = NanoLLM(
        vocab_size=C['vocab_size'], 
        d_model=d_model, 
        n_layer=n_layer, 
        n_head=n_head, 
        max_len=block_size,
        n_kv_head=n_kv_head,
        use_te=use_te
    )

    if use_te:
        model.to(device)
    else:
        model.to(device, dtype=torch.bfloat16)

    if os.path.exists(model_ckpt):
        print(f"Checkpoint found: {model_ckpt}. Loading weights...")
        ckpt = torch.load(model_ckpt, map_location='cpu')
        
        state_dict = ckpt['model_state_dict']
        
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("Detected compiled checkpoint, removing '_orig_mod.' prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully.")
    else:
        print("WARNING: No checkpoint was found")

    model.eval()
    return model, tokenizer

def generate_streaming(model, context, tokenizer, max_new_tokens=200, temperature=0.7, top_k=50, eos_token_id=None, stop_token_id=None, repetition_penalty=1.1, judge=None, current_user_prompt=""):
    generated_text = ""
    trigger_chars = [".", "!", "?", "\n"] 
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            block_size = getattr(model, 'max_len', 512)
            
            if context.shape[1] > block_size:
                idx_cond = context[:, -block_size:]
            else:
                idx_cond = context

            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if repetition_penalty > 1.0:
                logits = apply_repetition_penalty(logits, context, repetition_penalty, window=64)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
            
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                break
            
            token_text = tokenizer.decode([idx_next.item()])
            generated_text += token_text
            print(token_text, end='', flush=True)
            
            if judge is not None:
                in_dollar_math = generated_text.count("$") % 2 != 0
                in_code_block = generated_text.count("```") % 2 != 0
                
                latex_depth = (
                    generated_text.count("\\begin{") + generated_text.count("\\[") 
                    - generated_text.count("\\end{") - generated_text.count("\\]")
                )
                
                if in_code_block or in_dollar_math or latex_depth > 0 or generated_text.strip().endswith(":") or generated_text.strip().endswith("="):
                    pass 
                
                elif step > 5 and any(char in token_text for char in trigger_chars):
                    score = judge.predict([current_user_prompt, generated_text.strip()])

                    ratio = step / max_new_tokens
                    
                    current_threshold = max_thresh - (ratio * (max_thresh - min_thresh))
                    
                    if score > current_threshold:
                        print(f"\nDynamic stop! (Score: {score:.2f} > Threshold: {current_threshold:.2f} at step {step})", flush=True)
                        break

                if "\n" in token_text:
                    lines = [line.strip() for line in generated_text.split('\n') if len(line.strip()) > 4]
                    if len(lines) != len(set(lines)):
                        print("\nStop ! Repetition was detected...", flush=True)
                        break

            context = torch.cat((context, idx_next), dim=1)

    return generated_text

def get_multiline_input():
    bindings = KeyBindings()
    
    @bindings.add('escape', 'enter')
    def _(event):
        event.current_buffer.validate_and_handle()
    
    try:
        user_input = prompt(
            '👤 You: ',
            multiline=True,
            key_bindings=bindings,
            prompt_continuation='... '
        )
        return user_input
    except (EOFError, KeyboardInterrupt):
        return None

def start_chat(precision="bf16", model_ckpt=llm_chat_file_path, base_model=False, use_judge=False):
    model, tokenizer = load_loaded_model(precision, model_ckpt)
    
    judge = None
    if use_judge:
        print("Loading Semantic Judge (TinyBERT)...")
        from sentence_transformers import CrossEncoder
        judge = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512, device='cpu')
    
    try:
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    except:
        eos_token_id = None

    print("\n" + "="*50)
    print("🤖 NanoLLM Chat")
    print("TIP: Press Enter for new lines, Alt+Enter to send.")
    print("Type 'exit', 'quit', 'clear' (to reset memory), or 'q' to quit.")
    print("="*50 + "\n")

    messages = []

    while True:
        user_input = get_multiline_input()
        
        if user_input is None:
            break
        
        user_lower = user_input.strip().lower()
        if user_lower in ["exit", "quit", "q"]:
            break
        
        if user_lower == "clear":
            messages = []
            print("History cleared.\n")
            continue
            
        if not user_input.strip():
            continue

        if base_model:
            formatted_prompt = user_input
            current_user_prompt = user_input
            new_user_ids = tokenizer.encode(formatted_prompt)
        else:
            messages.append({"role": "user", "content": user_input})
            current_user_prompt = user_input
            
            if len(messages) > 3:
                messages = messages[-3:]
            
            while True:
                formatted_prompt = ""
                for msg in messages:
                    formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                
                formatted_prompt += "<|im_start|>assistant\n"
                new_user_ids = tokenizer.encode(formatted_prompt)
                
                if len(new_user_ids) <= max_prompt_len or len(messages) <= 1:
                    break
                else:
                    messages.pop(0)

        current_context = torch.tensor(new_user_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        print("\n🤖 AI: ", end='', flush=True)
        
        chat_stop_id = tokenizer.encode("<|im_end|>")[0] if not base_model else None
        
        full_generation = generate_streaming(
            model=model, 
            context=current_context,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens, 
            temperature=0.2,        
            top_k=30, 
            eos_token_id=eos_token_id,
            stop_token_id=chat_stop_id if not base_model else eos_token_id,
            repetition_penalty=1.1,
            judge=judge,
            current_user_prompt=current_user_prompt
        )
        
        if not base_model:
            messages.append({"role": "assistant", "content": full_generation})
        
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with NanoLLM")
    
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["bf16", "fp8"], 
        default=None,
        help="Precision mode: bf16 (PyTorch native) or fp8 (Transformer Engine). If not specified, uses config value."
    )

    parser.add_argument(
        "--ckpt", 
        type=str, 
        default=llm_chat_file_path, 
        help="path to the checkpoint",
    )

    parser.add_argument(
        "--base", 
        action='store_true', 
        help="Use base model mode (no chat format wrapper)",
    )
    
    parser.add_argument(
        "--judge", 
        action='store_true', 
        help="Activer le juge sémantique TinyBERT pour tronquer les réponses.",
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("Generation Configuration")
    print("="*50)
    print(f"Precision:        {args.precision.upper() if args.precision else 'CONFIG'}")
    print(f"Device:           {device}")
    print(f"Checkpoint:       {args.ckpt}")
    print(f"Chat format:      {'BASE' if args.base else 'CHATBOT'}")
    print(f"Semantic Judge:   {'ENABLED' if args.judge else 'DISABLED'}")
    print("="*50)
    
    start_chat(args.precision, args.ckpt, args.base, args.judge)