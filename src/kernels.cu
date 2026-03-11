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
    float min_value = fminf(norm_r, fminf(norm_g, norm_b));

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
    else if (norm_r == max_value) { 
        *h = ((norm_g - norm_b) / (max_value - min_value)) * 60;
    }
    else if (norm_g == max_value) {
        *h = ((norm_b - norm_r) / (max_value - min_value)) * 60;
    }
    else if (norm_b == max_value) {
        *h = ((norm_r - norm_g) / (max_value - min_value)) * 60;
    }
    else {
        printf("Error in calculating rbgToHsv conversion");
        return;
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
    
    //Mapping each thread to a pixel in the image
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    //If the thread is within the bounds of the image, perform the rbgToHsv conversion and write to the output array
    if (idx < width && idy < height) {

        //Finding the rbg values for the pixel
        int pixel_index = (idy * width + idx) * 3; // Each pixel has 3 components (R, G, B)
        unsigned char r = input_img[pixel_index];
        unsigned char g = input_img[pixel_index + 1];
        unsigned char b = input_img[pixel_index + 2];

        //Perform the rbgToHsv conversion
        float h, s, v;
        rbgToHsv(r, g, b, &h, &s, &v);

        //Write the hsv values to the output array
        output_img[pixel_index] = h;
        output_img[pixel_index + 1] = s;
        output_img[pixel_index + 2] = v;

    }

    return;
}

/**
 * A kernel that filters out the red pixels in an image. Each thread will map to a pixel and check if the red value is above a certain threshold. 
 * If it is, then the output pixel will be set to white, otherwise it will be set to black.The output will be a binary image where the red pixels 
 * are white and the non-red pixels are black. The input is an hsv image, so the red pixels will be determined by checking if the hue value is within 
 * a certain range (e.g. 0-10 degrees or 350-360 degrees) and if the saturation and value are above certain thresholds.
 * Each output hsv pixel will be converted back to rgb format and written to the output array as a binary image (white for red pixels, black for non-red pixels).
 */
__global__ void filterRed(float *input_img, unsigned char *output_img, int height, int width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height) {

        int pixel_index = (idy * width + idx) * 3; // Each pixel has 3 components (H, S, V)
        float h = input_img[pixel_index];
        float s = input_img[pixel_index + 1];
        float v = input_img[pixel_index + 2];

        // Check if the pixel is red based on hue, saturation, and value thresholds
        if ((h >= 0 && h <= 10) || (h >= 350 && h <= 360)) {

            if (s > 0.5f && v > 0.5f) {
                output_img[pixel_index / 3] = 255; // Set to white
            }
            else {
                output_img[pixel_index / 3] = 0; // Set to black
            }
            
        }
        else {
            output_img[pixel_index / 3] = 0; // Set to black
        }

    }

    return;

}
