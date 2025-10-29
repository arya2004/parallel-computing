#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorAdd(float  *A, float *B, float *C, int n){

    //index of thread in block. we are using  1 block only
    int i = threadIdx.x;

    if (i < n){
        C[i] = A[i] + B[i];
    }
    
}



int main(){

    int n = 20;
    size_t size = n * sizeof(float);

    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    for (int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = (i * i) - i;
    }
    

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c,d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++){
        cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    }
        
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return 0;
}