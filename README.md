# Spectral Phase Transitions in Neural Manifolds

**Unifying Random Matrix Theory (RMT) and Hessian Curvature for Zero-Shot Compression**

> **Author**: Chenmeng Wang  
> **Institution**: Imperial College London

## Abstract
The massive overparameterization of modern neural architectures necessitates efficient compression, yet current Low-Rank Approximation methods predominantly rely on empirical heuristics. In this repository, we introduce a unified physical framework that frames neural weight truncation as a predictable topological phase transition. 

By integrating **Random Matrix Theory (RMT)** to establish spectral noise limits with Hutchinson-based **Density-Aware Hessian Trace estimation** to map the thermodynamic curvature of the loss landscape, we precisely identify the "Structural Anchors" and "Holographic Reservoirs" within neural manifolds.

Extensive cross-modal evaluations across **Implicit Neural Representations (NeRF)**, **3D Transformers (MASt3R)**, and **Large Language Models (TinyLlama-1.1B, Qwen2.5-1.5B)** explicitly validate the universality of our framework.

## Core Discoveries
- **The Denoising Dividend**: In 3D geometric tasks (MASt3R), singularity-anchored truncation prevents feature annihilation and achieves an active **+0.96% accuracy gain** via targeted denoising of the decoder manifold, seamlessly translating to an **8.15% real-time hardware inference speedup**.
- **LLM Redundancy Core**: We successfully compress up to **45.1%** of Qwen2.5-1.5B's massive redundancy core without triggering a logical cliff.
- **The Geometric Knee**: In NeRFs, the framework accurately predicts the phase transition, preserving **98.5%** of spatial rendering fidelity with nearly half the parameters.

## Repository Structure
- `src/`: Core logic and experimental scripts.
  - `nerf/`: Density-aware Hessian trace and RMT analysis for spatial representations.
  - `llm/`: Perplexity (PPL) extreme panoramas and grid sweeps for language manifolds.
  - `vision/`: Batch-size throughput testing and 120-matrix full network decomposition for MASt3R.
- `results/`: High-resolution HPC grid sweep CSVs and generated topology reports.
- `docs/`: The full academic LaTeX report and associated high-fidelity figures.

## Usage
The code utilizes `jax` for highly optimized matrix-free Hessian-Vector Products (HVP) on NeRF architectures, and `PyTorch` for massive Transformer network (LLM and ViT) manipulations.

## License
MIT License
