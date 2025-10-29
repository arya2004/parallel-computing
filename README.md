
# Parallel Computing 

Works only on systems with NVIDIA GPU and proper CUDA drivers.

This repository contains CUDA/C++ source codes for all major practicals and assignments of the undergraduate Parallel Computing course, covering topics such as Amdahl’s Law, vector and matrix operations, shared memory, reduction, texture memory, image processing, and lightweight scientific applications.


---

## Installation (Ubuntu)

1. **Install CUDA Toolkit**

   ```bash
   sudo apt update
   sudo apt install nvidia-cuda-toolkit
   nvcc --version
   ```

2. **Install OpenCV (for image processing codes)**

   ```bash
   sudo apt install libopencv-dev
   pkg-config --modversion opencv4
   ```

3. **Install Build Essentials**

   ```bash
   sudo apt install build-essential cmake git pkg-config
   ```

---

## Compilation and Execution

Compile a CUDA file:

```bash
cd src
nvcc 02_vector_addition_1d.cu -o vector_add
./vector_add
```

Compile a CUDA file using OpenCV:

```bash
nvcc -std=c++17 06_image_processing.cu -o image_proc `pkg-config --cflags --libs opencv4`
./image_proc
```




