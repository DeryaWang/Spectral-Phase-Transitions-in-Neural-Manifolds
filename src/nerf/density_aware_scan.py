import jax
import jax.numpy as jnp
from jax import nn, grad, random, jit, vmap
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Configuration and Loading
MODEL_PATH = "/Users/kaying/Desktop/summer/nerf-master/logs/lego_example/model_200000.npy"
NEW_RESULT_DIR = "/Users/kaying/Desktop/summer/new_result/lego"
NUM_SAMPLES = 2048 # Increased for better manifold coverage
HESSIAN_VECTORS = 50 # Hutchinson vectors

def load_weights(path):
    raw_data = np.load(path, allow_pickle=True)
    return [jnp.array(p) for p in raw_data]

def positional_encoding(x, L=10):
    rets = [x]
    for i in range(L):
        for fn in [jnp.sin, jnp.cos]: rets.append(fn(2.**i * x))
    return jnp.concatenate(rets, axis=-1)

def nerf_forward(params, x_enc):
    h = x_enc
    for i in [0, 2, 4, 6, 8]: h = nn.relu(jnp.dot(h, params[i]) + params[i+1])
    h = jnp.concatenate([x_enc, h], axis=-1)
    h = nn.relu(jnp.dot(h, params[10]) + params[11])
    for i in [12, 14, 16]: h = nn.relu(jnp.dot(h, params[i]) + params[i+1])
    return jnp.dot(h, params[22]) + params[23] # This is Alpha/Density head

# 2. Density-Aware Loss and Trace
def compute_density_aware_trace(params, x_enc, y_true, sigma_weights, layer_idx):
    """
    Computes Trace(H) where Loss = sum(sigma * (y_true - y_pred)^2)
    """
    def loss_fn(W_target):
        new_params = list(params)
        new_params[layer_idx] = W_target
        y_pred = nerf_forward(new_params, x_enc)
        # Apply density weights
        sq_err = (y_pred - y_true)**2
        weighted_loss = jnp.mean(sigma_weights * sq_err)
        return weighted_loss

    target_W = params[layer_idx]
    
    @jit
    def hvp(v):
        return grad(lambda W: jnp.sum(grad(loss_fn)(W) * v))(target_W)

    key = random.PRNGKey(42)
    trace_sum = 0.0
    for _ in range(HESSIAN_VECTORS):
        key, subkey = random.split(key)
        v = random.choice(subkey, jnp.array([-1., 1.]), shape=target_W.shape)
        trace_sum += jnp.sum(v * hvp(v))
    
    return float(trace_sum / HESSIAN_VECTORS)

# 3. Main Sensitivity Scan
def run_density_aware_sensitivity():
    print("--- STEP 1: DENSITY-AWARE SENSITIVITY SCAN ---")
    params = load_weights(MODEL_PATH)
    
    # Sample points and get density weights
    key = random.PRNGKey(0)
    x_raw = random.uniform(key, (NUM_SAMPLES, 3), minval=-1.0, maxval=1.0)
    x_enc = vmap(positional_encoding)(x_raw)
    
    y_true = nerf_forward(params, x_enc)
    # sigma_weights: we use the raw density output as the weight. 
    # To avoid bias from negative logits, we apply softplus or just relu.
    # In NeRF, alpha = 1 - exp(-softplus(logit)). We'll use softplus(logit) as 'physical density'.
    sigma_weights = nn.softplus(y_true) 
    
    # Weight layers (indices from inspection)
    weight_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    layer_names = {i: f"Layer_{i}" for i in weight_layers}
    
    results = {}
    print(f"{'Layer':<10} | {'Trace (Density-Aware)':<20} | {'Rel. Sensitivity'}")
    print("-" * 55)
    
    for idx in weight_layers:
        trace = compute_density_aware_trace(params, x_enc, y_true, sigma_weights, idx)
        # Sensitivity = sqrt(Trace / Params)
        sens = np.sqrt(trace / params[idx].size)
        results[idx] = {"trace": trace, "sens": sens}
        print(f"{layer_names[idx]:<10} | {trace:20.6f} | {sens:.6f}")

    # Log to file
    report_path = os.path.join(NEW_RESULT_DIR, "sensitivity_report.txt")
    with open(report_path, "w") as f:
        f.write("DENSITY-AWARE HESSIAN SENSITIVITY REPORT (LEGO)\n")
        f.write("="*50 + "\n")
        for idx in weight_layers:
            f.write(f"{layer_names[idx]}: Trace={results[idx]['trace']:.4f}, Sensitivity={results[idx]['sens']:.6f}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    names = [layer_names[i] for i in weight_layers]
    sens_vals = [results[i]["sens"] for i in weight_layers]
    plt.bar(names, sens_vals, color='teal')
    plt.xticks(rotation=45)
    plt.ylabel("Relative Sensitivity (Density-Aware)")
    plt.title("Lego Model Layer Sensitivity Scan")
    plt.tight_layout()
    plt.savefig(os.path.join(NEW_RESULT_DIR, "layer_sensitivity.png"))
    
    max_idx = max(results, key=lambda k: results[k]["sens"])
    print(f"\n[WINNER] Target Layer identified: {layer_names[max_idx]}")
    return max_idx

if __name__ == "__main__":
    run_density_aware_sensitivity()
