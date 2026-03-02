# Kernel Image Processing with CUDA

## Project Overview
This project explores the acceleration of kernel-based image processing—a fundamental technique in computer vision and AI—by leveraging the parallel computing power of GPUs. While sequential CPU implementations often become a bottleneck as image resolutions increase, this study demonstrates how **CUDA (Compute Unified Device Architecture)** can be used to achieve significant speedups, often orders of magnitude faster than traditional methods.

The project compares a sequential CPU implementation with multiple CUDA-based parallel implementations, evaluating them based on execution time, computational complexity, and efficiency.


## Technologies & CUDA Implementations
The project utilizes **CUDA** to map individual pixel computations to separate GPU threads, allowing for massive parallelization. Several parallelization strategies were implemented to optimize performance:

* **1-Channel Based Parallelization**: Parallelizes execution within a single channel of the input image, building the final result by looping over the three RGB channels sequentially.
* **Constant Memory Optimization (`__constant__`)**: Utilizes CUDA's constant memory to store read-only variables and kernel values. This optimizes read operations via a dedicated cache and efficient broadcasting to multiple threads, significantly reducing memory latency.
* **3-Channel Based Parallelization**: Improves efficiency by linearizing the three-channel image structure into a single large array, reducing the overhead of repeated data transfers between the host and device.
* **Memory Coalescing**: Explores different thread and block organizations (grid vs. global index) to optimize how data is accessed in memory, which is critical for achieving high performance on the GPU.

## Methodologies: Scaling & Performance
The robustness and efficiency of the parallel implementations were evaluated using two primary experimental setups, serving as a form of **weak scaling** to test how the code handles increasing data sizes:

1. **Fixed Image Size (Scaling Kernel Size)**: The image resolution is fixed (e.g., 1000x1000) while the kernel size is increased. This highlights how parallel complexity savings grow quadratically with the kernel size.
2. **Fixed Kernel Size (Scaling Image Size)**: The kernel size is fixed (e.g., 11x11) while the image resolution is increased from 100x100 to 10000x100000. This test reveals the impact of data transfer overhead and linearization procedures on overall performance.

### Key Findings
* **Significant Speedup**: CUDA implementations consistently outperformed the sequential version, with speedups reaching over 500x in certain fixed-image tests.
* **Memory Management**: The use of `__constant__` memory proved crucial, providing a robust performance boost across different kernel sizes.
* **Bottlenecks**: Data transfer between the CPU and GPU (via PCIe) and sequential linearization steps on the CPU are the primary bottlenecks for very large images.

## System Specifications
* **CPU**: Dual-socket Intel Xeon Silver 4314 (32 physical cores).
* **GPU**: NVIDIA RTX A2000 12GB (3,328 Ampere CUDA Cores).
* **Language**: C++ with CUDA.
