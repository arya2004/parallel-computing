#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixMultiply(float *A, float *B, float *C, int width){

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < width && c < width){

        float sum = 0.0f;
        for (int i = 0; i < width; i++)
        {
            sum += A[r * c + i] * B[i * width + c];
        }
        

        C[r * width + c] = sum;

    }

}


int main(){

    int n = 4;

    size_t size = n * n * sizeof(float);


    float *h_a = new float[n * n];
    float *h_b = new float[n * n];
    float *h_c = new float[n * n];

    for (int i = 0; i < n * n; i++){
        h_a[i] = i;
        h_b[i] = (i * 0.7) - 0.4;
    }
    

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(16, 16);
    dim3 numBlocks((n + threadPerBlock.x - 1) / threadPerBlock.x , (n + threadPerBlock.y - 1) / threadPerBlock.y);

    matrixMultiply<<<numBlocks, threadPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c,d_c, size, cudaMemcpyDeviceToHost);

    cout << "Matrix A:\n";
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout << h_a[i * n + j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix B:\n";
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout << h_b[i * n + j] << " ";
        }
        cout << endl;
    }

    cout << "\n result:\n";
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            cout << h_c[i * n + j] << " ";
        }
        cout << endl;
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return 0;
}