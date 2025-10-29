#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void matrixAdd(float  *A, float *B, float *C, int r, int c){

    //index of thread in block. we are using  1 block only
    int i = threadIdx.x;
    int total = r * c;

    if (i < total){
        C[i] = A[i] + B[i];
    }
    
}



int main(){

    int r = 4;
    int c = 16;

    //total should be under 1024

    int total = r * c;
    size_t size = total * sizeof(float);

    float *h_a = new float[total];
    float *h_b = new float[total];
    float *h_c = new float[total];

    for (int i = 0; i < total; i++){
        h_a[i] = i;
        h_b[i] = (i * 2) - 1;
    }
    

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    matrixAdd<<<1, total>>>(d_a, d_b, d_c, r, c);

    cudaMemcpy(h_c,d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
        {
            cout << h_c[i * c + j] << " ";
        }
        cout << endl;
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return 0;
}