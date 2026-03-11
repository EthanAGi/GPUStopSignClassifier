#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__ float distance( int p1[], int p2[] );

__global__ void computeKernels( int *d_points, float *d_kernels, int numPoints, int numKernels );

__global__ void drawCircleKernel(unsigned char **pixels, int numRows, int numCols, int centerRow, int centerCol, float radius);

__global__ void checkShape(int *values, int *max, int *reg_maxes, int num_regions, int n);

__device__ void rbgToHsv(unsigned char r, unsigned char g, unsigned char b, float *h, float *s, float *v);

__global__ void rgbToHsvKernel(unsigned char *input_img, float *output_img, int height, int width);

__global__ void filterRed(int *values, int *max, int *reg_maxes, int num_regions, int n);

