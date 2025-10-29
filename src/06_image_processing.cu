#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

__global__ void rgbToGray(unsigned char *input, unsigned char *output, int w, int h){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < w && y < h){

        int idx = (y * w + x) * 3; //3 for rgb
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];

        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        output[y * w + x] = gray;

    }

}


int main()  {


    Mat image =  imread("input.jpg", IMREAD_COLOR);

    int w = image.cols;
    int h = image.rows;

    size_t colorSize = w * h * 3 * sizeof(unsigned char);
    size_t graySize = w * h * sizeof(unsigned char);

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, colorSize);
    cudaMalloc(&d_output, graySize);

    cudaMemcpy(d_input, image.data, colorSize, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((w + threadsPerBlock.x - 1) / threadsPerBlock.x, (h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    rgbToGray<<<numBlocks, threadsPerBlock>>>(d_input, d_output, w, h);

    Mat grayImage(h, w, CV_8UC1);
    cudaMemcpy(grayImage.data, d_output, graySize, cudaMemcpyDeviceToHost);

    imwrite("grey.jpg", grayImage);
    cout << "old schooled successfully" << endl;

    cudaFree(d_input);

    cudaFree(d_output);


    return 0;
}