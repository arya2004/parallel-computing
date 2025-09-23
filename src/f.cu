#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N (100000)
#define TPB 256  // threads per block

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c = (float*)malloc(N * sizeof(float));
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int gridSize = (N + TPB - 1) / TPB;
    cudaEventRecord(start, 0);
    vectorAdd<<<gridSize, TPB>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, start, stop);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t t0 = clock();
    for (int i = 0; i < N; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }
    clock_t t1 = clock();
    double cpuMs = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;

    printf("CPU sequential time:   %.3f ms\n", cpuMs);
    printf("GPU kernel time:       %.3f ms\n", gpuMs);
    if (gpuMs > 0.0f) {
        printf("Speedup (CPU/GPU):     %.2fx\n", cpuMs / gpuMs);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
