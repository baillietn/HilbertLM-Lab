import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

from config import config

target_tokens_B = config['pre_training_target_tokens'] / 1e9
sft_target_tokens_B = config['stf_target_tokens'] / 1e9

def load_checkpoint_history(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}...")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except FileNotFoundError:
        print(f"File not found: {checkpoint_path}")
        return None

    if 'history' not in ckpt:
        print(f"This checkpoint does not contain history: {checkpoint_path}")
        return None
    
    return ckpt['history']

def plot_training(pretrain_ckpt, sft_ckpt=None, output_file="training_report.png", sft_stretch_factor=5):
    pretrain_history = load_checkpoint_history(pretrain_ckpt)
    if pretrain_history is None:
        return
    
    tokens = list(np.array(pretrain_history['tokens']) / 1e9)
    loss = list(pretrain_history['loss'])
    lr = list(pretrain_history['lr'])
    
    if sft_ckpt:
        sft_history = load_checkpoint_history(sft_ckpt)
        if sft_history is not None:
            last_pretrain_tokens = tokens[-1] if tokens else 0
            sft_tokens = np.array(sft_history['tokens']) / 1e9
            sft_tokens_stretched = sft_tokens * sft_stretch_factor
            
            tokens.extend(sft_tokens_stretched + last_pretrain_tokens)
            loss.extend(sft_history['loss'])
            lr.extend(sft_history['lr'])
            print(f"Combined histories: {len(pretrain_history['tokens'])} steps (pretrain) + {len(sft_history['tokens'])} steps (SFT) [stretch x{sft_stretch_factor}]")
    
    tokens = np.array(tokens)
    loss = np.array(loss)
    lr = np.array(lr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    
    sft_end = target_tokens_B + (sft_target_tokens_B * sft_stretch_factor) if sft_ckpt else target_tokens_B + sft_target_tokens_B
    
    stages = [
        (0, 0.4 * target_tokens_B, "Stage 1\n(General: 85% Web)", 'blue'),
        (0.4 * target_tokens_B, 0.7 * target_tokens_B, "Stage 2\n(+Knowledge: 10% Cosmo)", 'cyan'),
        (0.7 * target_tokens_B, 0.9 * target_tokens_B, "Stage 3\n(+Reasoning: 15% Math)", 'orange'),
        (0.9 * target_tokens_B, target_tokens_B, "Stage 4\n(Expert: 25% Cosmo + 20% Math)", 'red'),
        (target_tokens_B, sft_end, "Stage 5\n(SFT: 75% Chat)", 'green') 
    ]

    window = 50
    if len(loss) > window:
        loss_smooth = np.convolve(loss, np.ones(window)/window, mode='valid')
        x_smooth = tokens[window-1:]
        ax1.plot(x_smooth, loss_smooth, color='#2c3e50', linewidth=1.5, label='Smoothed Loss')
        ax1.plot(tokens, loss, color='#2c3e50', alpha=0.1, label='Raw Loss')
    else:
        ax1.plot(tokens, loss, color='#2c3e50')
    
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.set_title("Training Dynamics (Curriculum Learning: 4 Stages + SFT)", fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True, fancybox=True, shadow=True)

    y_min, y_max = ax1.get_ylim()
    for start, end, name, color in stages:
        ax1.axvspan(start, end, color=color, alpha=0.05)
        if start < tokens[-1]:
            mid_point = (start + end) / 2
            ax1.text(mid_point, y_max - (y_max-y_min)*0.05, name, 
                    color=color, fontweight='bold', ha='center', fontsize=9)
    
    if sft_ckpt:
        ax1.axvline(x=target_tokens_B, color='green', linestyle='--', linewidth=2, alpha=0.7, label='SFT Start')
        real_sft_end = target_tokens_B + sft_target_tokens_B
        ax1.axvline(x=tokens[-1], color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.text(tokens[-1], y_max - (y_max-y_min)*0.15, 
                f'Real SFT end\n{real_sft_end:.2f}B tokens', 
                color='purple', fontweight='bold', ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    ax2.plot(tokens, lr, color='#e74c3c', linewidth=2)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_xlabel("Tokens processed (Billions)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    for start, end, _, color in stages:
        ax2.axvspan(start, end, color=color, alpha=0.05)
    
    if sft_ckpt:
        ax2.axvline(x=target_tokens_B, color='green', linestyle='--', linewidth=2, alpha=0.7)
        real_sft_end = target_tokens_B + sft_target_tokens_B
        ax2.axvline(x=tokens[-1], color='purple', linestyle=':', linewidth=1.5, alpha=0.6)
        ax2.text(tokens[-1], ax2.get_ylim()[1] * 0.95, 
                f'{real_sft_end:.2f}B', 
                color='purple', fontweight='bold', ha='center', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Graph generated: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="checkpoints/final_base_model.pt", 
                        help="Path to pretraining checkpoint")
    parser.add_argument("--sft", type=str, default="checkpoints/nano_llm_chat_final.pt",
                        help="Path to SFT checkpoint (optional)")
    parser.add_argument("--output", type=str, default="figures/training_report.png",
                        help="Output file")
    parser.add_argument("--stretch", type=int, default=5,
                        help="SFT phase stretch factor (default: 5)")
    
    args = parser.parse_args()
    
    plot_training(args.pretrain, args.sft, args.output, args.stretch)