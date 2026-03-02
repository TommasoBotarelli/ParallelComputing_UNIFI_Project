# High-Performance Parallel Computing Suite

This repository serves as a comprehensive portfolio of Parallel Computing course held by professor Marco Bertini at the University of Florence. 

The focus is on optimizing computationally intensive tasks across different hardware architectures. From low-level GPU kernel optimization to high-level Python multiprocessing, these projects demonstrate a deep understanding of bottlenecks, scaling laws, and hardware-aware programming.

---

## 🚀 Projects Overview

**Note**: for a comprehensive overview of each experiment see the `README.md` files in each folder.

### 1. Kernel Image Processing with CUDA
**Focus:** GPU Acceleration & Memory Hierarchy Optimization
* **Objective:** Accelerate fundamental computer vision kernels using NVIDIA's Parallel Thread Execution.
* **Key Implementations:**
    * Optimized memory access using `__constant__` memory for filter kernels.
    * Comparison of 1-Channel vs. 3-Channel linearized parallelization to reduce Host-to-Device overhead.
    * Memory coalescing strategies to maximize PCIe bandwidth.
* **Impact:** Achieved speedups of over **500x** compared to sequential C++ implementations.
* **Tech:** C++, CUDA, NVIDIA RTX A2000.
* **Addtional info:** [link](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/tree/master/kernel_image_processing)

### 2. Time Series Pattern Recognition with OpenMP
**Focus:** Shared-Memory Parallelism & NUMA Awareness
* **Objective:** Solve the Subsequence Matching problem (identifying patterns in massive datasets) using optimized distance metrics.
* **Key Implementations:**
    * Manual domain decomposition using `omp_get_thread_num()` to outperform standard `#pragma` directives.
    * Strategic thread binding (`OMP_PROC_BIND=spread`) to maximize memory bandwidth on dual-socket systems.
    * Reduction clauses and pre-checks to eliminate thread contention in critical sections.
* **Tech:** C++, OpenMP, Intel Xeon Silver (Dual-Socket).
* **Additional info:** [link](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/tree/master/pattern_recognition)

### 3. Image Augmentation Pipeline (Python Multiprocessing)
**Focus:** Distributed CPU Workloads & I/O Bottleneck Mitigation
* **Objective:** Optimize data preprocessing pipelines for Deep Learning by bypassing the Python Global Interpreter Lock (GIL).
* **Key Implementations:**
    * **Architecture $P_2$:** Independent worker reads to reduce IPC (Inter-Process Communication) overhead.
    * **Architecture $P_Q$:** Producer-Consumer model with dedicated "saver" processes to mitigate disk I/O latency.
* **Analysis:** Investigated Strong vs. Weak scaling and identified the "superlinear" effect of cache optimization for resolutions $\le 1000 \times 1000$.
* **Tech:** Python, Albumentations, OpenCV, Multiprocessing.
* **Addtional info:** [link](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/tree/master/image_augmentation)

---

## 📊 Performance Engineering & Methodology

Every project in this repository follows a rigorous scientific benchmarking process:

| Metric | Description |
| :--- | :--- |
| **Strong Scaling** | Keeping problem size fixed while increasing processing units (Ranks/Threads) to identify the serial bottleneck (Amdahl's Law). |
| **Weak Scaling** | Proportional increase of workload and resources to test system efficiency at scale (Gustafson's Law). |
| **Hardware Profiling** | Analysis of Memory Bus Contention, Cache Thrashing, and PCIe latency. |


---

## 🛠 Hardware Environments

Experiments were conducted on high-performance workstations to simulate production environments:
* **CPU:** Dual-socket Intel Xeon Silver 4314 (32 Physical Cores / 64 Threads).
* **GPU:** NVIDIA RTX A2000 (12GB VRAM, 3,328 CUDA Cores).
* **OS:** Ubuntu server.

---

## 📂 Repository Structure

```text
.
├── kernel_image_processing/    # C++/CUDA source & PDF Reports
├── pattern_recognition/        # C++/OpenMP source & PDF Reports
├── image_augmentation/         # Python source & Scaling Analysis
├── presentazioni/              # Presentations
└── resports/                   # PDF reports
