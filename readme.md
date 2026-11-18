# CPU-And-GPU-Test: PyTorch GPU vs. CPU Deep Learning Stress Test & Hardware Telemetry Benchmark

**Repository**: [https://github.com/parthraorane86/CPU-And-GPU-Test](https://github.com/parthraorane86/CPU-And-GPU-Test)

A hardware-agnostic Jupyter Notebook benchmark (`gpu_training_stress_test.ipynb`) that verifies whether your Python / VS Code / Conda environment is truly utilizing your **dedicated NVIDIA GPU** during deep learning model training—rather than silently falling back to your CPU or getting confused with your laptop's integrated graphics in Windows Task Manager.

Unlike simple `torch.cuda.is_available()` checks or tiny matrix multiplications that barely use 500 MB of memory, this benchmark trains an **adaptive Deep Residual Neural Network** (forward pass, Cross-Entropy loss, backward gradient propagation, and `AdamW` weight updates) while sampling live hardware telemetry from `nvidia-smi`, PyTorch memory counters, and system CPU threads.

## Why This Benchmark Exists

1. **Silent CPU Fallbacks**: Installing the default `pip install torch` on Windows installs the `+cpu` wheel, causing training scripts to run 15x–30x slower on CPU without throwing an error.

2. **Task Manager Dual-GPU Confusion**: On hybrid laptops (Intel/AMD iGPU + NVIDIA dGPU), Windows Task Manager defaults to showing `GPU 0` (integrated graphics) and the `3D` rendering graph rather than the `Cuda` compute graph, making it look like the dedicated GPU is idle even when CUDA is running.

3. **Asynchronous CUDA Queue Freezes**: Naive `while time.time() - t < 15:` GPU stress tests without `torch.cuda.synchronize()` inside the loop stuff hundreds of kernels into the asynchronous CUDA queue, freezing Jupyter notebooks for minutes.

4. **100% Hardware-Agnostic & Auto-Scaling**: Works automatically on **any NVIDIA GPU** (RTX 20/30/40/50 series, GTX 16 series, or datacenter GPUs from 4 GB to 80 GB VRAM) and **any CPU** across Windows, Linux, and macOS.

## Key Features

* **Automatic Hardware Discovery**: Detects your exact CPU model, physical/logical core counts, system RAM, secondary integrated GPU (and its reserved RAM on Windows), and your dedicated NVIDIA GPU's architecture, Streaming Multiprocessors (SMs), exact CUDA core count, and total VRAM.

* **VRAM Auto-Scaling**: Automatically scales the Deep Residual Neural Network dimensions (`3072` to `6144`), residual blocks, and pre-loaded VRAM batch cache to allocate **~65%–75% of your specific GPU's VRAM** (whether you have 4 GB, 6 GB, 8 GB, 12 GB, or 24 GB+ of VRAM).

* **Stage 1 (CPU Baseline)**: Forces training onto `device='cpu'` to record baseline throughput (`samples/sec`), active CPU thread usage, and verify `0 MB` of GPU training VRAM.

* **Stage 2 (15-Second Full-Load GPU Stress Test)**: Trains the scaled neural network on `device='cuda:0'` with per-batch synchronization (`torch.cuda.synchronize()`) while printing real-time telemetry every 2 seconds:

  * Active GPU Compute Load (`%`) and estimated active CUDA cores

  * Dedicated VRAM Used (`MiB`) and PyTorch Peak Allocated VRAM (`MB`)

  * Real-time GPU Power Draw (`Watts`) and Core Temperature (`°C`)

* **Automated Side-by-Side Comparison & Zero-Leak Cleanup**: Prints a comparison table and a plain-English Pass/Fail summary tailored to your detected hardware, then runs garbage collection (`gc.collect()` + `torch.cuda.empty_cache()`) so your GPU returns to `0 MB` allocated VRAM after the test.

## Sample Output (Tested on NVIDIA GeForce RTX 4050 6GB Laptop GPU)

```text
============================================================================================
 DETAILED COMPARISON: CPU TRAINING vs. NVIDIA GEFORCE RTX 4050 LAPTOP GPU TRAINING
============================================================================================
Metric                             | Stage 1: CPU Only       | Stage 2: NVIDIA GeForce RT  
--------------------------------------------------------------------------------------------
Compute Hardware Used              | 16 CPU Threads          | 20 SMs (2,560 CUDA Cores)   
Neural Network Size                | 35.7M Parameters        | 222.3M Parameters (6.2x)    
Peak GPU CUDA Core Load (%)        | 19% (0 Training Cores)  | 100% (~2,560 Cores)         
PyTorch Peak Training VRAM         | 0 MB                    | 4,507 MB (4.40 GB)          
Total Dedicated GPU Memory Used    | Idle / Display Only     | 5,653 / 6,140 MiB           
Peak GPU Power Draw                | 9.8 W                   | 79.7 W                      
Training Throughput (Speed)        | 738.6 samples/s         | 2,744.1 samples/s (3.7x)    
============================================================================================
```

## Prerequisites

* **Python**: `3.9+` (tested on Python `3.11`)

* **NVIDIA Driver**: Installed with `nvidia-smi` accessible in your system PATH

* **Packages**:

  * `torch` (CUDA-enabled build from [pytorch.org](https://pytorch.org/get-started/locally/))

  * `psutil` (for CPU physical/logical core and RAM telemetry)

## Quick Start & Steps to Run

### 1. Clone the Repository

```bash
git clone https://github.com/parthraorane86/CPU-And-GPU-Test.git
cd CPU-And-GPU-Test
```

### 2. Install CUDA-Enabled PyTorch & `psutil`

Make sure your virtual or Conda environment has a **CUDA build** of PyTorch (not the default `+cpu` build):

```bash
# Activate your environment (if using Conda)
conda activate ds

# Install psutil and CUDA-enabled PyTorch (example for CUDA 12.1+)
pip install psutil
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run the Notebook in VS Code or Jupyter

1. Open `CPU And GPU Test.ipynb` in **VS Code** or **Jupyter Lab / Notebook**.

2. Select your GPU-enabled Python kernel in the top-right kernel picker.

3. Click **Restart** and then **Run All** to execute all 5 cells from top to bottom (~25 seconds total runtime).

## Troubleshooting Common Issues

| Symptom in Output | Cause | Fix |
| :--- | :--- | :--- |
| `CUDA Available: False` and PyTorch ends in `+cpu` | CPU-only PyTorch wheel is installed | Run `pip uninstall -y torch torchvision torchaudio` and reinstall using the official CUDA `--index-url` from [pytorch.org](https://pytorch.org/get-started/locally/) |
| `nvidia-smi` not found | NVIDIA GPU driver is missing or not on PATH | Update your driver via the NVIDIA App / GeForce Experience |
| Windows Task Manager shows `0%` on GPU during Stage 2 | Task Manager defaults to showing `GPU 0` (integrated GPU) or the `3D` graph | In Task Manager -> **Performance**, click **GPU 1** (your NVIDIA GPU) and switch the graph dropdown from **3D** to **Cuda** |
