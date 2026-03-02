# Time Series Pattern Recognition with OpenMP

## Project Overview
This project addresses the computationally intensive task of **Time Series Pattern Recognition**, specifically focusing on the **Subsequence Matching** problem. The goal is to identify a specific pattern (a smaller time series of length *m*) within a much larger time series (length *N*) by minimizing a distance metric. 

The project evaluates various implementation strategies, ranging from a baseline sequential algorithm to highly optimized parallel versions using **OpenMP**.

* [Presentation](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/blob/master/presentazioni/pattern_recognition.pdf)
* [PDF report](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/blob/master/reports/pattern_recognition.pdf)

## Methodologies
The core of the project involves a rigorous performance analysis using two primary scaling methodologies:

* **Strong Scaling**: Measures performance by keeping the problem size fixed while increasing the number of threads (from 1 to 64). This test helps identify how much the execution time decreases as more resources are added.
* **Weak Scaling**: Measures performance by increasing both the problem size and the number of threads proportionally, keeping the workload per thread constant. This demonstrates how well the solution handles larger datasets.

## Technologies & Implementations
The project utilizes **OpenMP** to implement several parallelization strategies within a shared-memory paradigm:

### Distance Metric
* **Sum of Absolute Differences (SAD)**: Used as the primary metric to calculate the distance between the reference pattern and subsequences.

### Implementation Levels
1.  **Sequential (Baseline)**: A standard nested-loop approach (Algorithm 1).
2.  **Simple Parallel**: Uses basic `#pragma omp parallel for` directives (Algorithm 2).
3.  **Smart Parallel**: Enhances the simple version by optimizing the critical section using the `reduction(min:minDistance)` clause and pre-checks to reduce thread contention (Algorithm 4).
4.  **Manual Parallel**: The most performant version, which manually divides the problem domain among threads using `omp_get_thread_num()` and `omp_get_num_threads()` to minimize overhead (Algorithm 3).

## Key Findings & Enhancements
* **Manual Optimization**: The "Manual" implementation significantly outperformed standard OpenMP directives, especially as thread counts reached the system's physical limits.
* **Thread Affinity**: Performance is heavily influenced by thread binding (`OMP_PROC_BIND`). The **spread** configuration often yielded the best results by distributing threads across physical cores and sockets, thereby maximizing memory bandwidth and reducing cache contention.
* **Architecture Awareness**: The study highlights that on multi-socket NUMA systems, memory utilization and socket-to-socket latency are critical factors in achieving high efficiency.

## Experimental Environment
* **Hardware**: Dual-socket Intel Xeon Silver 4314 (32 physical cores, 64 logical threads).
* **Data Generation**: Synthetic time series data created using the `mockseries` Python library.
