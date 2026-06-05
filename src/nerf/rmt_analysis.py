import jax
import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt

def load_weights(model_path):
    raw_data = np.load(model_path, allow_pickle=True)
    params = [jnp.array(p) for p in raw_data]
    return params

def marchenko_pastur_analysis(W, name, output_dir):
    """
    Implements Marchenko-Pastur filtering to separate Signal from Noise.
    """
    m, n = W.shape
    # Ensure Q >= 1
    if m < n:
        W = W.T
        m, n = n, m
    
    Q = m / n
    # Eigenvalues of the covariance matrix C = (1/n) * W * W.T (size m x m)
    # or C = (1/m) * W.T * W (size n x n). Let's use the smaller one for eigenvalues.
    C = (W.T @ W) / m
    eigvals = jnp.linalg.eigvalsh(C)
    eigvals = jnp.maximum(eigvals, 1e-10)
    
    # Estimate sigma^2 (noise variance) from the bulk
    # A robust estimator is the median of the eigenvalues
    sigma2 = jnp.median(eigvals) / 1.0 # Simple heuristic, can be refined
    
    # MP Upper Bound: lambda+ = sigma^2 * (1 + sqrt(1/Q))^2
    lambda_plus = sigma2 * (1 + jnp.sqrt(1/Q))**2
    lambda_minus = sigma2 * (1 - jnp.sqrt(1/Q))**2
    
    # 1. Identify Spikes (Signal)
    spikes = eigvals[eigvals > lambda_plus]
    num_spikes = len(spikes)
    
    # 2. Reconstruct Entropy on Spikes ONLY
    p_signal = spikes / jnp.sum(spikes)
    h_signal = -jnp.sum(p_signal * jnp.log(p_signal + 1e-12))
    r_eff_true = jnp.exp(h_signal)
    
    print(f"\n--- RMT Analysis: {name} ---")
    print(f"Matrix Shape: {m}x{n} (Q={Q:.2f})")
    print(f"Est. Noise Sigma^2: {sigma2:.6f}")
    print(f"MP Upper Bound (λ+): {float(lambda_plus):.6f}")
    print(f"Number of Signal Spikes: {num_spikes}")
    print(f"TRUE Effective Rank (R_eff_true): {float(r_eff_true):.2f}")
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    
    # Histogram of all eigenvalues
    plt.hist(eigvals, bins=100, density=True, alpha=0.3, color='gray', label='Empirical Spectrum')
    
    # Plot MP Density Curve for the bulk
    x = jnp.linspace(float(lambda_minus), float(lambda_plus), 100)
    def mp_pdf(val):
        return Q / (2 * jnp.pi * sigma2) * jnp.sqrt((lambda_plus - val) * (val - lambda_minus)) / val
    
    plt.plot(x, [mp_pdf(v) for v in x], color='red', linewidth=2, label='Marchenko-Pastur Bulk')
    
    # Mark the Spikes
    plt.scatter(spikes, jnp.zeros_like(spikes), color='blue', s=20, label='Signal Spikes', zorder=5)
    plt.axvline(x=lambda_plus, color='red', linestyle='--', label='λ+ (Threshold)')
    
    plt.title(f"RMT Spectrum Analysis: {name} (Signal vs. Noise)", fontsize=15)
    plt.xlabel("Eigenvalue (λ)")
    plt.ylabel("Density")
    plt.yscale('log')
    plt.legend()
    
    plot_path = os.path.join(output_dir, f"rmt_spectrum_{name.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    return float(r_eff_true), num_spikes

def run_rmt_fusion():
    # Analyze Lego
    print("\n" + "="*50)
    print("ANALYZING LEGO DATASET")
    print("="*50)
    lego_path = "Desktop/summer/nerf-master/logs/lego_example/model_200000.npy"
    lego_params = load_weights(lego_path)
    lego_output = "Desktop/summer/svd_experiment/results_Lego"
    os.makedirs(lego_output, exist_ok=True)
    
    # Analyze Lego Layer 10 and Layer 16
    r_true_lego_10, n_spikes_lego_10 = marchenko_pastur_analysis(lego_params[10], "Lego Layer 10", lego_output)
    r_true_lego_16, n_spikes_lego_16 = marchenko_pastur_analysis(lego_params[16], "Lego Layer 16", lego_output)
    
    # Analyze Fern
    print("\n" + "="*50)
    print("ANALYZING FERN DATASET")
    print("="*50)
    fern_path = "Desktop/summer/nerf-master/logs/fern_example/model_200000.npy"
    fern_params = load_weights(fern_path)
    fern_output = "Desktop/summer/svd_experiment/results_Fern"
    os.makedirs(fern_output, exist_ok=True)
    
    # Analyze Layer 10 and 16 for Fern
    r_true_fern_10, n_spikes_fern_10 = marchenko_pastur_analysis(fern_params[10], "Fern Layer 10", fern_output)
    r_true_fern_16, n_spikes_fern_16 = marchenko_pastur_analysis(fern_params[16], "Fern Layer 16", fern_output)

    print(f"\n[GLOBAL RMT SUMMARY]")
    print(f"Lego Layer 10: True Eff Rank = {r_true_lego_10:.2f}, Spikes = {n_spikes_lego_10}")
    print(f"Lego Layer 16: True Eff Rank = {r_true_lego_16:.2f}, Spikes = {n_spikes_lego_16}")
    print(f"Fern Layer 10: True Eff Rank = {r_true_fern_10:.2f}, Spikes = {n_spikes_fern_10}")
    print(f"Fern Layer 16: True Eff Rank = {r_true_fern_16:.2f}, Spikes = {n_spikes_fern_16}")


if __name__ == "__main__":
    run_rmt_fusion()
