import argparse
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from tqdm import tqdm

from model import NanoLLM
from config import config

device = config['device']
dtype = torch.bfloat16
grid_steps = 25
morph_frames = 10
pre_training_target_tokens = config['pre_training_target_tokens']

def get_stage_from_tokens(tokens_count):
        
    if tokens_count < 0.4 * pre_training_target_tokens: return 1
    if tokens_count < 0.7 * pre_training_target_tokens: return 2
    if tokens_count < 0.9 * pre_training_target_tokens: return 3
    if tokens_count < pre_training_target_tokens: return 4
    
    return 4

def load_validation_batches():
    print("Loading validation batches from .bin files...")
    dest = config['dataset_path']
    seq_len = config['block_size'] + 1
    batches = {}
    
    for stage in range(1, 6):
        data_path = os.path.join(dest, f"val_stage_{stage}_data.bin")
        labels_path = os.path.join(dest, f"val_stage_{stage}_labels.bin")
        
        if not os.path.exists(data_path):
            print(f"Missing file: {data_path}. Make sure you have run get_data.py --validation")
            continue
            
        inputs = np.fromfile(data_path, dtype=np.uint16).astype(np.int64)
        labels = np.fromfile(labels_path, dtype=np.int32).astype(np.int64)
        
        inputs = torch.tensor(inputs, dtype=torch.long, device=device).view(-1, seq_len)
        labels = torch.tensor(labels, dtype=torch.long, device=device).view(-1, seq_len)
        
        batches[stage] = {"inputs": inputs, "labels": labels}
        
    return batches

def flatten_weights(state_dict):
    return torch.cat([p.flatten().to('cpu', dtype=dtype) for p in state_dict.values()])

def unflatten_weights(flat_weights, reference_state_dict):
    new_dict = {}
    offset = 0
    for name, param in reference_state_dict.items():
        numel = param.numel()
        new_dict[name] = flat_weights[offset:offset+numel].view(param.shape)
        offset += numel
    return new_dict

def compute_pca_and_gram_schmidt(checkpoints_flat):
    print("Computing PCA and orthonormalization on CPU...")
    theta_star = checkpoints_flat[-1]
    
    X = torch.stack([ckpt - theta_star for ckpt in checkpoints_flat])
    
    Gram = torch.matmul(X, X.T) 
    eigenvalues, eigenvectors = torch.linalg.eigh(Gram.float())
    
    v1 = eigenvectors[:, -1].to(dtype)
    v2 = eigenvectors[:, -2].to(dtype)
    
    delta_1 = torch.matmul(v1, X)
    delta_2 = torch.matmul(v2, X)
    
    print("Applying strict Gram-Schmidt...")
    delta_1 = delta_1 / torch.norm(delta_1)
    
    projection = torch.dot(delta_1, delta_2) * delta_1
    delta_2 = delta_2 - projection
    delta_2 = delta_2 / torch.norm(delta_2)
    
    dot_product = torch.dot(delta_1, delta_2).item()
    print(f"Orthogonality verified (Dot product: {dot_product:.2e})")
    
    return theta_star, delta_1, delta_2

@torch.no_grad()
def evaluate_loss_surfaces(model, theta_star, d1, d2, validation_batches, ref_dict, alphas, betas):
    print(f"Computing Loss Landscapes on {grid_steps}x{grid_steps} grid...")
    
    surfaces = {stage: np.zeros((grid_steps, grid_steps)) for stage in validation_batches.keys()}
    micro_batch_size = 4
    
    for i, alpha in enumerate(tqdm(alphas, desc="Balayage Alpha")):
        for j, beta in enumerate(betas):
            mutant_flat = theta_star + (alpha * d1) + (beta * d2)
            model.load_state_dict(unflatten_weights(mutant_flat, ref_dict))
            
            for stage, batch in validation_batches.items():
                inputs_full = batch["inputs"][:, :-1]
                labels_full = batch["labels"][:, 1:]
                
                stage_loss = 0.0
                n_samples = inputs_full.size(0)
                
                for k in range(0, n_samples, micro_batch_size):
                    inputs = inputs_full[k:k+micro_batch_size]
                    labels = labels_full[k:k+micro_batch_size]
                    
                    logits = model(inputs)
                    shift_logits = logits.contiguous().view(-1, logits.size(-1))
                    shift_labels = labels.contiguous().view(-1)
                    
                    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
                    
                    stage_loss += loss.item() * (inputs.size(0) / n_samples)
                    
                surfaces[stage][i, j] = stage_loss
                
    return surfaces

def get_z_altitude(alpha_val, beta_val, alphas_arr, betas_arr, surface):
    idx_a = np.abs(alphas_arr - alpha_val).argmin()
    idx_b = np.abs(betas_arr - beta_val).argmin()
    return surface[idx_a, idx_b]

def create_smooth_animation(alphas, betas, surfaces, proj_coords, ckpt_stages):
    print("Generating animation with smooth morphing...")
    A, B = np.meshgrid(alphas, betas)
    
    all_z = np.concatenate([s.flatten() for s in surfaces.values()])
    valid_z = all_z[np.isfinite(all_z)]
    
    if len(valid_z) == 0:
        print("NaN or infinite values detected in loss surfaces. Using default z-limits.")
        z_min, z_max = 0.0, 10.0
    else:
        z_min = valid_z.min()
        z_max = np.percentile(valid_z, 95)
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    writer = PillowWriter(fps=10)
    
    def draw_frame(current_surface, active_idx, title, stage_color):
        ax.clear()
        ax.set_zlim(z_min, z_max)
        
        ax.plot_surface(A, B, current_surface.T, cmap='viridis', edgecolor='none', alpha=0.6)
        
        history_x = [p[0] for p in proj_coords[:active_idx+1]]
        history_y = [p[1] for p in proj_coords[:active_idx+1]]
        history_z = [get_z_altitude(x, y, alphas, betas, current_surface) for x, y in zip(history_x, history_y)]
        
        ax.plot(history_x, history_y, history_z, color='white', linestyle='--', linewidth=2)
        
        if active_idx > 0:
            ax.scatter(history_x[:-1], history_y[:-1], history_z[:-1], color='silver', s=40, alpha=0.7)
            
        ax.scatter([history_x[-1]], [history_y[-1]], [history_z[-1]], color=stage_color, s=250, marker='*', edgecolor='black')
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("First principal direction (PCA)")
        ax.set_ylabel("Second principal direction (PCA)")
        ax.set_zlabel("Cross-Entropy Loss")
        
        ax.view_init(elev=30, azim=45) 
        writer.grab_frame()

    colors = {1: 'cyan', 2: 'yellow', 3: 'orange', 4: 'red', 5: 'lime'}
    
    with writer.saving(fig, "figures/curriculum_smooth_landscape.gif", dpi=120):
        prev_stage = ckpt_stages[0]
        
        for idx in range(len(proj_coords)):
            target_stage = ckpt_stages[idx]
            
            if target_stage != prev_stage:
                print(f"  -> Morphing: Stage {prev_stage} to Stage {target_stage}...")
                surf_prev = surfaces[prev_stage]
                surf_target = surfaces[target_stage]
                
                for t in np.linspace(0, 1, morph_frames):
                    morphed_surface = (1 - t) * surf_prev + t * surf_target
                    title = f"Curriculum Transition: Stage {prev_stage} ➔ {target_stage}"
                    draw_frame(morphed_surface, idx - 1, title, 'white')
                    
            title = f"Training : Stage {target_stage} (Checkpoint {idx+1}/{len(proj_coords)})"
            for _ in range(5): 
                draw_frame(surfaces[target_stage], idx, title, colors[target_stage])
                
            prev_stage = target_stage

    print("Animation completed: figures/curriculum_smooth_landscape.gif")

def main(args):
    precision = args.precision if args.precision else ("fp8" if config['use_te'] else "bf16")
    use_te = (precision == "fp8")
    
    model = NanoLLM(
        vocab_size=config['vocab_size'], 
        d_model=config['d_model'], 
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        n_kv_head=config['n_kv_head'], 
        max_len=config['block_size'],
        use_te=use_te
    )

    model.to(device, dtype=torch.bfloat16)
    
    val_batches = load_validation_batches()
    
    print("Reading checkpoints...")
    
    ckpt_files = glob.glob("checkpoints/samples/ckpt_*.pt")
    sft_files = glob.glob("checkpoints/samples/sft_ckpt_*.pt")
    
    if sft_files:
        ckpt_files.extend(sft_files)
    else:
        print("Warning: No SFT checkpoint found with this pattern.")
        
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint was found...")
        
    ckpt_files.sort(key=os.path.getmtime)
    
    checkpoints_flat = []
    ckpt_stages = []
    ref_dict = None
    
    for path in ckpt_files:
        print(f"  -> Loading {os.path.basename(path)} on CPU...")
        ckpt = torch.load(path, map_location='cpu')
        
        history = ckpt.get('history', {})
        tokens_list = history.get('tokens', [])
        tokens_count = tokens_list[-1] if tokens_list else 0
            
        stage = get_stage_from_tokens(tokens_count)
        
        if "sft_ckpt" in os.path.basename(path):
            stage = 5
            
        ckpt_stages.append(stage)
        
        state_dict = ckpt.get('model_state_dict', ckpt)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
        checkpoints_flat.append(flatten_weights(state_dict))
        if ref_dict is None: ref_dict = state_dict
        
    print(f"Detected stage sequence: {ckpt_stages}")

    theta_star, d1, d2 = compute_pca_and_gram_schmidt(checkpoints_flat)
    
    print("Projecting all checkpoints...")
    proj_coords = []
    for ckpt_flat in checkpoints_flat:
        diff = ckpt_flat - theta_star
        alpha = torch.dot(diff, d1).item()
        beta = torch.dot(diff, d2).item()
        proj_coords.append((alpha, beta))
        
    alphas_proj = [p[0] for p in proj_coords]
    betas_proj = [p[1] for p in proj_coords]
    
    margin_a = (max(alphas_proj) - min(alphas_proj)) * 0.2
    margin_b = (max(betas_proj) - min(betas_proj)) * 0.2
    margin_a = margin_a if margin_a > 0 else 1.0
    margin_b = margin_b if margin_b > 0 else 1.0
    
    min_a, max_a = min(alphas_proj) - margin_a, max(alphas_proj) + margin_a
    min_b, max_b = min(betas_proj) - margin_b, max(betas_proj) + margin_b
    
    alphas = np.linspace(min_a, max_a, grid_steps)
    betas = np.linspace(min_b, max_b, grid_steps)
    
    theta_star = theta_star.to(device)
    d1 = d1.to(device)
    d2 = d2.to(device)
    
    surfaces = evaluate_loss_surfaces(model, theta_star, d1, d2, val_batches, ref_dict, alphas, betas)
    create_smooth_animation(alphas, betas, surfaces, proj_coords, ckpt_stages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss landscape")
    
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["bf16", "fp8"], 
        default="bf16",
        help="Precision mode: bf16 (PyTorch native) or fp8 (Transformer Engine)"
    )

    args = parser.parse_args()

    main(args)