import sys
import os
import argparse
import shutil

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)

if parent_folder not in sys.path:
    sys.path.append(parent_folder)

import torch
from transformers import AutoTokenizer
from config import config
from modeling import HilbertLMConfig, HilbertLMForCausalLM

def adapt_state_dict_for_hf(state_dict, use_layernorm):
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if "amax" in key or "scale_inv" in key or "_extra_state" in key:
            continue
            
        if not use_layernorm:
            if "layer_norm_bias" in key or "ln1.bias" in key or "ln2.bias" in key or "final_norm.bias" in key:
                continue
            
        try:
            if value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                value = value.to(torch.bfloat16)
        except AttributeError:
            pass 
            
        value = value.to(torch.bfloat16)

        if "ln_attn.layer_norm_weight" in key:
            new_key = key.replace("ln_attn.layer_norm_weight", "ln1.weight")
        elif "ln_attn.layer_norm_bias" in key:
            new_key = key.replace("ln_attn.layer_norm_bias", "ln1.bias")
        elif "ln_attn.weight" in key:
            new_key = key.replace("ln_attn.weight", "qkv_proj.weight")
        elif "ln_mlp.layer_norm_weight" in key:
            new_key = key.replace("ln_mlp.layer_norm_weight", "ln2.weight")
        elif "ln_mlp.layer_norm_bias" in key:
            new_key = key.replace("ln_mlp.layer_norm_bias", "ln2.bias")
        elif "ln_mlp.fc1_weight" in key:
            new_key = key.replace("ln_mlp.fc1_weight", "mlp.0.weight")
        elif "ln_mlp.fc2_weight" in key:
            new_key = key.replace("ln_mlp.fc2_weight", "mlp.2.weight")
        else:
            new_key = key 
            
        new_key = "model." + new_key
        new_state_dict[new_key] = value
        
    return new_state_dict


def detect_architecture(state_dict, config_dict):
    """Detect model architecture from checkpoint state dict."""
    use_te = any("ln_attn" in k or "ln_mlp" in k for k in state_dict.keys())
    use_layernorm = any("ln1.bias" in k or "layer_norm_bias" in k or "final_norm.bias" in k for k in state_dict.keys())
    
    d_model = state_dict["token_embedding.weight"].shape[1]
    mlp_key = "layers.0.ln_mlp.fc1_weight" if use_te else "layers.0.mlp.0.weight"
    
    if mlp_key in state_dict:
        mlp_out_features = state_dict[mlp_key].shape[0]
        use_swiglu = (mlp_out_features != 4 * d_model)
    else:
        use_swiglu = config_dict['use_swiglu']
        
    if "lm_head.weight" not in state_dict:
        tie_weights = True
    else:
        tie_weights = torch.equal(state_dict["lm_head.weight"], state_dict["token_embedding.weight"])
    
    return use_te, use_layernorm, use_swiglu, tie_weights

def main(args):
    print(f"Loading checkpoint from: {args.ckpt}...")

    if not os.path.isfile(args.ckpt):
        print(f"Error: Checkpoint file '{args.ckpt}' not found.")
        return
    
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    raw_state_dict = checkpoint.get('model_state_dict', checkpoint)
    raw_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}

    print("\n-> Architecture analysis...")
    use_te, use_layernorm, use_swiglu, tie_weights = detect_architecture(raw_state_dict, config)
    
    print(f"   * Transformer Engine : {'Enabled' if use_te else 'Disabled'}")
    print(f"   * Normalization      : {'LayerNorm' if use_layernorm else 'RMSNorm'}")
    print(f"   * MLP Activation     : {'SwiGLU' if use_swiglu else 'GELU'}")
    print(f"   * Weight Tying       : {'Enabled' if tie_weights else 'Disabled'}\n")
    
    print("Adapting state dict for Hugging Face...")
    clean_state_dict = adapt_state_dict_for_hf(raw_state_dict, use_layernorm)
    
    if tie_weights and "model.lm_head.weight" in clean_state_dict:
        del clean_state_dict["model.lm_head.weight"]
    
    print("Creating Hugging Face configuration...")
    hf_config = HilbertLMConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['d_model'],
        num_hidden_layers=config['n_layer'],
        num_attention_heads=config['n_head'],
        num_key_value_heads=config['n_kv_head'],
        block_size=config['block_size'],
        use_layernorm=use_layernorm,
        use_swiglu=use_swiglu,
        tie_word_embeddings=tie_weights,
    )
    
    model = HilbertLMForCausalLM(hf_config)
    
    print("Injecting weights...")
    model.load_state_dict(clean_state_dict, strict=False)
    
    print("Converting to bfloat16...")
    model = model.to(torch.bfloat16)
    
    if tie_weights:
        model.tie_weights()
        output_emb = model.get_output_embeddings()
        input_emb = model.get_input_embeddings()
        
        if output_emb.weight is input_emb.weight:
            print("✓ Weight tying verified")
        else:
            print("✗ WARNING: Weight tying failed!")
    
    print("Saving model...")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    HilbertLMConfig.register_for_auto_class()
    HilbertLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
    state_dict_to_save = model.state_dict()
    if tie_weights and "model.lm_head.weight" in state_dict_to_save:
        state_dict_to_save = {k: v for k, v in state_dict_to_save.items() if k != "model.lm_head.weight"}
    
    model.save_pretrained(output_dir, safe_serialization=True, state_dict=state_dict_to_save)
    
    modeling_source = os.path.join(os.path.dirname(__file__), "modeling.py")
    modeling_dest = os.path.join(output_dir, "modeling.py")
    shutil.copy(modeling_source, modeling_dest)
    
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Completed! Model ready in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom HilbertLM checkpoint to Hugging Face format")
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="checkpoints/hilbert_chat_model.pt", 
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="hf_export_model",
        help="Output directory for the converted model"
    )
    args = parser.parse_args()
    main(args)