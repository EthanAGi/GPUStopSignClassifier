#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ float distance( int p1[], int p2[] )
{
    return sqrtf( (float)( (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) ) );

}

__global__ void drawCircleKernel(unsigned char **pixels, int numRows, int numCols, int centerRow, int centerCol, float radius) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < numRows && col < numCols) {
        int p[2] = {row, col};
        int center[2] = {centerRow, centerCol};
        float dist = distance(p, center);
        if (fabs(dist - radius) <= 0.5) {
            pixels[row][col] = 255; // Set pixel to white if it's on the circle
        }
    }

}

__global__ void checkShape(int *values, int *max, int *reg_maxes, int num_regions, int n) {
    return;
}

__global__ void rbgToHsv(int *values, int *max, int *reg_maxes, int num_regions, int n) {

    return;

}

__global__ void filterRed(int *values, int *max, int *reg_maxes, int num_regions, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    return;

}

