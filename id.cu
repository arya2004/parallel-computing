#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void printInfo() {

    int myBlockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

    int myBlockOffset = myBlockId * blockDim.x * blockDim.y * blockDim.z;
    
    int threadOffset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int id = myBlockOffset + threadOffset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",id, blockIdx.x, blockIdx.y, blockIdx.z, myBlockId, threadIdx.x, threadIdx.y, threadIdx.z, threadOffset);

    printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}


int main(){

    const int bx = 2;
    const int by = 3;
    const int bz = 4;

    const int tx = 1;
    const int ty = 1;
    const int tz = 1;

    int blockPerGrid = bx * by * bz;
    int threadPerBlock = tx * ty * tz;

    printf("%d blocks/grid\n", blockPerGrid);
    printf("%d threads/block\n", threadPerBlock);
    printf("%d total threads\n", blockPerGrid * threadPerBlock);


    dim3 bPG(bx, by, bz);
    dim3 tPB(tx, ty, tz);

    printInfo<<<bPG, tPB>>>();

    cudaDeviceSynchronize();




    return 0;
}