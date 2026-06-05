import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import os
import matplotlib.pyplot as plt

MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
RESULTS_DIR = '/Users/kaying/Desktop/summer/llama_experiment/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def apply_svd(W, rank):
    rank = int(round(float(rank)))
    if rank <= 0: return torch.zeros_like(W)
    max_r = min(W.shape)
    rank = min(rank, max_r)
    U, S, V = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    S_trunc = S.clone(); S_trunc[rank:] = 0
    return (U * S_trunc) @ V

def evaluate_ppl(model, tokenizer):
    # Using the exact same dataset, but limiting eval batches for local speed
    test = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    text_data = [t for t in test['text'] if len(t) > 10]
    encodings = tokenizer('\n\n'.join(text_data), return_tensors='pt')
    max_length = 512; stride = 256; seq_len = encodings.input_ids.size(1)
    
    # Use MPS if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    nlls = []
    prev_end_loc = 0
    # Process 20 chunks for a robust local estimate
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone(); target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len or len(nlls) > 20: break 
    return torch.exp(torch.stack(nlls).sum() / end_loc).item()

def run_panorama():
    print('--- TINYLLAMA L2 PPL PANORAMA SWEEP (LOCAL MPS) ---')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval()

    L2_RMT_REFF = 198.4

    print('Calculating Baseline PPL...')
    with torch.no_grad():
        base_ppl = evaluate_ppl(model, tokenizer)
    print(f'Baseline PPL: {base_ppl:.4f}')

    layer = model.model.layers[2].mlp.down_proj
    W_orig = layer.weight.detach().clone()

    multipliers = np.concatenate([
        np.arange(0.1, 1.0, 0.1),
        np.arange(1.0, 3.0, 0.2),
        np.arange(3.0, 10.5, 1.0)
    ])

    results = []
    print(f'\n{"Multiplier":<10} | {"Rank":<8} | {"PPL":<10}')
    print('-'*35)

    csv_path = os.path.join(RESULTS_DIR, 'tinyllama_l2_ppl_panorama.csv')
    with open(csv_path, 'w') as f:
        f.write('Multiplier,Rank,PPL\n')
        f.write(f'Baseline,Full,{base_ppl:.4f}\n')

    for m in multipliers:
        rank = int(round(L2_RMT_REFF * m))
        with torch.no_grad():
            layer.weight.data = apply_svd(W_orig, rank).to(device)
            curr_ppl = evaluate_ppl(model, tokenizer)
        
        print(f'{m:<10.1f} | {rank:<8d} | {curr_ppl:<10.4f}')
        results.append((m, curr_ppl))
        with open(csv_path, 'a') as f:
            f.write(f'{m:.1f},{rank},{curr_ppl:.4f}\n')

    layer.weight.data = W_orig.to(device)

    # PLOTTING
    ms, ppls = zip(*results)
    
    # Create two plots: One full, one capped
    
    # 1. Capped Plot (to see the fine details)
    plot_ms, plot_ppls = [], []
    for m, p in zip(ms, ppls):
        if p < 50: # Cap Y axis
            plot_ms.append(m)
            plot_ppls.append(p)

    plt.figure(figsize=(10, 6))
    plt.plot(plot_ms, plot_ppls, 'r-o', linewidth=2.5, markersize=8, label='WikiText-2 PPL')
    plt.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='1.0x RMT Limit')
    plt.axhline(y=base_ppl, color='black', linestyle=':', alpha=0.6, label='Full-Rank Baseline')
    
    plt.xlabel('RMT Multiplier ($M$)', fontsize=12)
    plt.ylabel('Perplexity (PPL)', fontsize=12)
    plt.title('TinyLlama-1.1B L2: Absolute PPL Phase Transition', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path_capped = os.path.join(RESULTS_DIR, 'llama_l2_ppl_extreme_panorama_capped.png')
    plt.savefig(save_path_capped, dpi=300)
    plt.close()
    
    # 2. Log Scale Plot (to see the true abyss)
    plt.figure(figsize=(10, 6))
    plt.plot(ms, ppls, 'r-o', linewidth=2.5, markersize=8, label='WikiText-2 PPL')
    plt.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='1.0x RMT Limit')
    plt.axhline(y=base_ppl, color='black', linestyle=':', alpha=0.6, label='Full-Rank Baseline')
    
    plt.xlabel('RMT Multiplier ($M$)', fontsize=12)
    plt.ylabel('Perplexity (PPL - Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.title('TinyLlama-1.1B L2: The Sub-Critical Abyss (Log Scale)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    save_path_log = os.path.join(RESULTS_DIR, 'llama_l2_ppl_extreme_panorama_log.png')
    plt.savefig(save_path_log, dpi=300)
    plt.close()
    
    print(f'\n[SUCCESS] Saved PPL panorama plots to {RESULTS_DIR}')

if __name__ == "__main__":
    run_panorama()
