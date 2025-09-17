#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 25

__global__ void vectorAdd(float *a, float *b, float *c) {
    int idx = threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    float h_a[N] = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25 };
    float h_b[N] = { 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                     15, 14, 13, 12, 11, 10,  9,  8,  7,  6,
                      5,  4,  3,  2,  1 };
    float h_c[N];
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    vectorAdd<<<1, N>>>(d_a, d_b, d_c);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuMs = 0.0f;
    cudaEventElapsedTime(&gpuMs, start, stop);
    printf("GPU kernel time:   %.3f ms\n", gpuMs);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t cpuStart = clock();
    for (int i = 0; i < N; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }
    clock_t cpuEnd = clock();
    double cpuMs = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU serial time:   %.3f ms\n", cpuMs);

    for (int i = 0; i < N; i++) {
        printf("%.0f + %.0f = %.0f\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
