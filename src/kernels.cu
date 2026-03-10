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

/** RBG to HSV requires converting from the Red, Green, Blue model to Hue, Saturation, and Value.
 *  In order to convert from RGB to HSV, we must first normalize RGB values to a 0 - 1 range.
 *  Find the maximum and minimums to find the Value and Saturation. And then we can calculate the
 *  Hue by basing it on which color component is dominant.
 * 
 *  This method will take in the rgb values and write the results to the float pointers of hsv
 */
__device__ void rbgToHsv(unsigned char r, unsigned char g, unsigned char b, float *h, float *s, float *v) {

    //Normalize the values to a 0 - 1 range

    float norm_r = r / 255.0f;
    float norm_g = g / 255.0f;
    float norm_b = b / 255.0f;

    //Finding the value - the maximum value of r, g, and b
    float max_value = fmaxf(norm_r, fmaxf(norm_g, norm_b));
    *v = max_value;

    //Find min and (max - min) for future reference
    float min_value = fminf(norm_r, fminf(norm_f, norm_b));

    //Finding the Saturation - ((max - min) / max) OR 0 if the max is 0
    if (max_value == 0) {
        *s = 0.0f;
    }
    else {
        *s = (max_value - min_value) / max_value;
    }

/**
 * Calulating the Hue depending on what color is the max value:
 * 
 * If max - min is 0 then Hue == 0. This is a base case
 * 
 * Red is max, ((G - B) / (max - min)) * 60 degrees
 * 
 * Green is max, ((B - R) / (max - min)) * 60 degrees
 * 
 * Blue is max, ((R - G) / (max - min)) * 60 degrees
 */
    if (max_value - min_value == 0) {
        *h = 0.0f;
    }
    else {

        switch (max_value) {

        case norm_r:
            ((norm_g - norm_b) / (max_value - min_value)) * 60;
            break;

        case norm_g:
            ((norm_b - norm_r) / (max_value - min_value)) * 60;
            break;

        case norm_b:
            ((norm_r - norm_g) / (max_value - min_value)) * 60;
            break;

        default:
            printf("Error in calculating rbgToHsv conversion");
            return -1;

        }

    }
    
    if (*h < 0.0f) {
        *h += 360.0f;
    } 

}

/**
 *  A kernel that utilizes the rbgToHsv device method to convert all the rgb values into hsv at the same time using parallel computing.
 *  Each thread will map to each rbg pixel and perform the rbgToHsv kernel to find the hsv values for each pixel. The output will be an
 *  array of floats representing the converted hsv array of the image.
 */
__global__ void rgbToHsvKernel(unsigned char *input_img, float *output_img, int height, int width) {
    return;
}

__global__ void filterRed(int *values, int *max, int *reg_maxes, int num_regions, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    return;

}

