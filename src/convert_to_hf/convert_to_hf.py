import sys
import os
import argparse

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

def main(args):
    print(f"Loading checkpoint from: {args.ckpt}...")

    if not os.path.isfile(args.ckpt):
        print(f"Error: Checkpoint file '{args.ckpt}' not found.")
        return
    
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    
    raw_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    raw_state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw_state_dict.items()}

    print("\n-> Architecture analysis...")
    
    use_te = any("ln_attn" in k or "ln_mlp" in k for k in raw_state_dict.keys())
    
    use_layernorm = any("ln1.bias" in k or "layer_norm_bias" in k or "final_norm.bias" in k for k in raw_state_dict.keys())
    
    d_model = raw_state_dict["token_embedding.weight"].shape[1]
    mlp_key = "layers.0.ln_mlp.fc1_weight" if use_te else "layers.0.mlp.0.weight"
    if mlp_key in raw_state_dict:
        mlp_out_features = raw_state_dict[mlp_key].shape[0]
        use_swiglu = (mlp_out_features != 4 * d_model)
    else:
        use_swiglu = config['use_swiglu']
        
    if "lm_head.weight" not in raw_state_dict:
        tie_weights = True
    else:
        tie_weights = torch.equal(raw_state_dict["lm_head.weight"], raw_state_dict["token_embedding.weight"])

    print(f"   * Transformer Engine : {'Enabled' if use_te else 'Disabled'}")
    print(f"   * Normalisation      : {'LayerNorm' if use_layernorm else 'RMSNorm'}")
    print(f"   * Activation MLP     : {'SwiGLU' if use_swiglu else 'GELU'}")
    print(f"   * Weight Tying       : {'Enabled' if tie_weights else 'Disabled'}\n")
    
    print("Adapting state dict for Hugging Face...")
    clean_state_dict = adapt_state_dict_for_hf(raw_state_dict, use_layernorm)
    
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
        torch_dtype="bfloat16",
    )
    
    model = HilbertLMForCausalLM(hf_config)
    
    print("Injecting weights...")
    model.load_state_dict(clean_state_dict, strict=True)
    
    print("Saving in .safetensors and HF format...")
    output_dir = "hf_export_model"
    os.makedirs(output_dir, exist_ok=True)
    
    HilbertLMConfig.register_for_auto_class()
    HilbertLMForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
    if tie_weights:
        model.model.lm_head.weight = torch.nn.Parameter(model.model.token_embedding.weight.clone())

    model = model.to(torch.bfloat16)
    
    model.save_pretrained(output_dir, safe_serialization=True)
    
    print("Copying the Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    tokenizer.save_pretrained(output_dir)
    
    print(f"Completed ! Model ready for HF deployment in folder : {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom HilbertLM to Hugging Face format")

    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="checkpoints/hilbert_chat_model.pt", 
        help="Path to the checkpoint.pt file"
    )
    args = parser.parse_args()
    
    main(args)