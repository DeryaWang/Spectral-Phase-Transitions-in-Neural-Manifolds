import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# 1. Path setup
BASE = '/rds/general/user/cw2025/home/manifold_research/scripts/mast3r'
sys.path.append(BASE)
sys.path.append(os.path.join(BASE, 'dust3r'))
from mast3r.model import AsymmetricMASt3R

def analyze_rmt(W_np):
    m, n = W_np.shape
    if m < n: W_np = W_np.T; m, n = n, m
    Q = m / n
    C = (W_np.T @ W_np) / m
    eigvals = np.linalg.eigvalsh(C)
    sigma2 = np.median(eigvals)
    lambda_plus = sigma2 * (1 + np.sqrt(1/Q))**2
    spikes = eigvals[eigvals > lambda_plus]
    return np.exp(-np.sum((spikes/np.sum(spikes)) * np.log(spikes/np.sum(spikes) + 1e-12))) if len(spikes) > 0 else 0

def apply_svd_truncation(W, rank):
    rank = int(round(float(rank)))
    if rank <= 0: return torch.zeros_like(W)
    U, S, V = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    S_trunc = S.clone(); S_trunc[rank:] = 0
    return (U * S_trunc) @ V

def calculate_savings(orig_shape, rank):
    out_f, in_f = orig_shape
    orig_p = out_f * in_f
    comp_p = rank * (in_f + out_f)
    return (orig_p - comp_p) / orig_p * 100

def run_final_grid():
    print('--- MASt3R FINAL FIDELITY-SAVINGS GRID SWEEP ---')
    device = 'cuda'
    
    # Load ranking
    ranking = []
    with open('/rds/general/user/cw2025/home/manifold_research/speedup_test/results/mast3r_grand_sensitivity_report.txt', 'r') as f:
        for line in f: ranking.append(line.split(':')[0].strip())
    
    model = AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(device)
    model.eval()

    # Get Baseline Points
    v1 = {'img': torch.randn(1, 3, 512, 512).cuda(), 'true_shape': torch.tensor([[512, 512]]).cuda(), 'instance': torch.tensor([0]).cuda()}
    v2 = {'img': torch.randn(1, 3, 512, 512).cuda(), 'true_shape': torch.tensor([[512, 512]]).cuda(), 'instance': torch.tensor([1]).cuda()}
    with torch.no_grad():
        res_base, _ = model(v1, v2)
        base_pts = res_base['pts3d'].detach().clone()

    # Pre-cache original weights and RMT reffs
    print("Caching 120 matrices...")
    weight_cache = {}
    rmt_cache = {}
    for name in ranking:
        parts = name.split('_')
        block = model.enc_blocks[int(parts[1][1:])] if 'Enc' in name else model.dec_blocks[int(parts[1][1:])]
        if 'fc1' in name: mod = block.mlp.fc1
        elif 'fc2' in name: mod = block.mlp.fc2
        else: mod = (block.cross_attn if 'cross' in name else block.attn).proj
        
        W = mod.weight.detach().clone()
        weight_cache[name] = W
        rmt_cache[name] = analyze_rmt(W.cpu().numpy())

    results_file = '/rds/general/user/cw2025/home/manifold_research/speedup_test/results/mast3r_final_accuracy_grid.csv'
    with open(results_file, 'w') as f:
        f.write('Multiplier,N,Error,LayerSavings\n')

    for m in [2.5, 3.0, 3.5, 4.0]:
        print(f'\nSweeping Multiplier M={m}...')
        for n in range(20, 130, 10):
            targets = ranking[:n]
            total_saved_p = 0
            total_orig_p = 0
            
            with torch.no_grad():
                # Apply current grid config
                for name in ranking:
                    parts = name.split('_')
                    block = model.enc_blocks[int(parts[1][1:])] if 'Enc' in name else model.dec_blocks[int(parts[1][1:])]
                    if 'fc1' in name: mod = block.mlp.fc1
                    elif 'fc2' in name: mod = block.mlp.fc2
                    else: mod = (block.cross_attn if 'cross' in name else block.attn).proj
                    
                    W_orig = weight_cache[name]
                    total_orig_p += W_orig.numel()
                    
                    if name in targets:
                        rank = min(int(round(rmt_cache[name] * m)), min(W_orig.shape)-1)
                        mod.weight.copy_(apply_svd_truncation(W_orig, rank))
                        total_saved_p += (W_orig.numel() - (W_orig.shape[0] + W_orig.shape[1]) * rank)
                    else:
                        mod.weight.copy_(W_orig)

                # Measure 3D Error
                res_comp, _ = model(v1, v2)
                error = torch.norm(base_pts - res_comp['pts3d']).item() / torch.norm(base_pts).item() * 100
                savings = (total_saved_p / total_orig_p) * 100
                
                print(f'  N={n} | Error: {error:.4f}% | TargetSavings: {savings:.2f}%')
                with open(results_file, 'a') as f:
                    f.write(f'{m},{n},{error:.4f},{savings:.2f}\n')

if __name__ == "__main__":
    run_final_grid()
