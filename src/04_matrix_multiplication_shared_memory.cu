#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define TILE_WIDTH 16

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int width,  int tileWidth){

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int r = blockIdx.y * tileWidth + threadIdx.y;
    int c = blockIdx.x * tileWidth + threadIdx.x;

    float tempSum = 0.0f;
    int numTiles = (width + tileWidth - 1 ) / tileWidth;

    for (int i = 0; i < numTiles; i++)
    {
        if(r < width && (i * tileWidth + threadIdx.x) < width){
            tileA[threadIdx.y][threadIdx.x] = A[r * width + i * tileWidth + threadIdx.x];
        }else{
            tileA[threadIdx.y][threadIdx.x] =0.0f;
        }

        if(c < width && (i * tileWidth + threadIdx.y) < width){
            tileB[threadIdx.y][threadIdx.x] = B[(i * tileWidth + threadIdx.y) * width + c];
        }else{
            tileB[threadIdx.y][threadIdx.x] =0.0f;
        }

        __syncthreads();


        //multiply partials

        for (int j = 0; j < tileWidth; j++)
        {
            tempSum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();

    }

    if(r < width && c < width){
        C[r * width + c] = tempSum;
    }
    

}


int main(){

    int tileWidth = 16;

    int n = 4;

    size_t size = n * n * sizeof(float);


    float *h_a = new float[n * n];
    float *h_b = new float[n * n];
    float *h_c = new float[n * n];

    for (int i = 0; i < n * n; i++){
        h_a[i] = i;
        h_b[i] = (i * 0.4) - 0.7;
    }
    

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((n + tileWidth - 1) / tileWidth , (n + tileWidth - 1) / tileWidth);

    matrixMultiplyShared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n, tileWidth);

    cudaMemcpy(h_c,d_c, size, cudaMemcpyDeviceToHost);


    
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