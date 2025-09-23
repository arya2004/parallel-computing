
# Contributing to Parallel Computing Course Code



## 📋 What can you contribute?

- **New practicals:** Submit your implementation of a course-relevant CUDA or C++ parallel algorithm.
- **Improvements:** Optimize existing kernels, add performance tests, or improve code clarity.
- **Utilities:** Add helpful scripts or input generators for better testing.
- **Bug fixes:** Help squash bugs in existing code.
- **Test cases:** Add input/output files to help others validate code correctness.

**Note:**  
This repository is code-only (no theory, notes, or lecture slides).

---

## 🧑‍💻 Getting Started

1. **Fork this repository**  
   Click "Fork" at the top-right of the [repo page](https://github.com/arya2004/parallel-computing).

2. **Clone your fork locally**
    ```bash
    git clone https://github.com/<your-username>/parallel-computing.git
    cd parallel-computing
    ```

3. **Create a new branch**
    ```bash
    git checkout -b <feature-or-fix-name>
    ```

4. **Add your code**  
   - Place all new practicals in a new folder under `practicals/XX_name/`.
   - Each practical should have:  
     - `main.cu` (or appropriately named `.cu` file)  
     - `Makefile` for building  
     - `run.sh` script if needed

---

## ✏️ Code Style Guidelines

- **Language:** All code must be in C, C++, or CUDA C++ (`.cu`).
- **Naming:** Use descriptive folder and file names (`matmul_shared`, `reduction`, `sobel`).
- **Formatting:**  
  - Use consistent indentation (4 spaces preferred)
  - Comment your code for clarity (especially kernel launches and memory ops)
  - Use `snake_case` for file and folder names
- **Makefiles:** Each practical should include a `Makefile` to build its code using `nvcc`.
- **Testable:** If possible, include a minimal test case or input.

---

## 🔁 Pull Request Process

1. **Commit your changes**
    ```bash
    git add .
    git commit -m "Add practical: tiled shared memory matrix multiplication"
    ```

2. **Push to your fork**
    ```bash
    git push origin <feature-or-fix-name>
    ```

3. **Open a Pull Request**
   - Go to your fork on GitHub.
   - Click "Compare & Pull Request".
   - Provide a **clear title** and **description** of what your code does.
   - Reference related issues if any.

---


