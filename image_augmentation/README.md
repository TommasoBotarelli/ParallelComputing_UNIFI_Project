# Image Augmentation with Python Multiprocessing

This project explores the optimization of image augmentation pipelines using parallel computing techniques. Image augmentation is a vital technique in computer vision for artificially expanding datasets to improve model robustness and reduce overfitting. However, sequential processing of large-scale, high-resolution datasets often becomes a computational bottleneck.

## Core Technologies
* **Python Multiprocessing**: Leveraged to achieve true parallel execution across multiple CPU cores, bypassing the limitations of the Global Interpreter Lock (GIL).
* **Albumentations**: A high-level Python library used to create flexible augmentation pipelines involving geometric and photometric transformations.
* **OpenCV (cv2)**: Used for high-performance image reading, color space conversion, and writing operations.

* [Presentation](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/blob/master/presentazioni/image_augmentation.pdf)
* [PDF report](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project/blob/master/reports/image_augmentation.pdf)
---

## Methodology
The project implements and compares three distinct parallel architectures to identify the most efficient approach for handling large volumes of image data:

### Parallel Implementations
* **Standard Parallel ($P_1$)**: Images are read once by the main process, and the data is distributed to workers using the `multiprocessing.Pool` class.
* **Multiple Read ($P_2$)**: To reduce data transfer overhead, each worker process independently reads the required image files directly from the disk.
* **Queue-based Saving ($P_Q$)**: A specialized architecture where a subset of "saver" processes is dedicated to pulling transformed images from a shared queue and writing them to disk, aiming to mitigate I/O latency.

### Scaling Experiments

To evaluate performance, two types of scaling tests were conducted:
* **Strong Scaling**: Analyzes how execution time decreases as more processing units (ranks) are added to a fixed workload.
* **Weak Scaling**: Evaluates how the system scales when both the number of processes and the data size (image resolution) increase proportionally.

---

## Key Insights & Results
The experimentation revealed several critical findings regarding parallel image processing:

* **CPU Bound**: Results indicate the process is primarily CPU-bound, with CPU utilization reaching 100% during execution.
* **Memory Bus Contention**: Efficiency decays as the number of processes increases (visible at 8+ processes) due to memory bus saturation and cache thrashing.
* **Resolution Impact**: For images $\le 1000 \times 1000$, the trend can be superlinear due to cache optimization. Beyond this, efficiency drops significantly due to bandwidth contention.
* **I/O Trade-offs**: While $P_Q$ was designed to help with saving latency, the overhead of data transmission between processes prevented it from providing the expected performance gains.

---

## Future Developments
Future work involves deeper profiling to minimize process "waste time" and investigating advanced process distribution strategies to fully utilize available hardware resources.

---
*Author: Tommaso Botarelli*

*Project Link: [GitHub Repository](https://github.com/TommasoBotarelli/ParallelComputing_UNIFI_Project)*
