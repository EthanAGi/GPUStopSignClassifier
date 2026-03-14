#include "cpu_kernels.h"
#include <math.h>
#include <stdio.h>

#define GAUSSIAN_KERNEL_SIZE 3
// Defines gaussian kernel size as 3
#define GAUSSIAN_NORM 16.0f
// Normalization factor for Gaussian kernel

static const float gaussianKernel[GAUSSIAN_KERNEL_SIZE][GAUSSIAN_KERNEL_SIZE] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f}
};
// 3x3 Gaussian kernel to smooth binary mask.
// Center pixel has highest weight, surrounding pixels contribute less

static int clampInt(int value, int minVal, int maxVal) {
    // Clamps an int into a valid range
    if (value < minVal) return minVal;
    // If too small, return min value
    if (value > maxVal) return maxVal;
    // If too large, return max value
    return value;
    // Returns value
}

static void rgbToHsvSingle(unsigned char r, unsigned char g, unsigned char b,
                           float *h, float *s, float *v) {
                            // Starts a helper function which converts RGB to HSV
    float norm_r = r / 255.0f;
    float norm_g = g / 255.0f;
    float norm_b = b / 255.0f;
    // Normalizes RGB values to [0, 1] range

    float max_value = fmaxf(norm_r, fmaxf(norm_g, norm_b));
    float min_value = fminf(norm_r, fminf(norm_g, norm_b));
    // Finds the max and min values needed for HSV Conversion

    *v = max_value;
    // Value (V) is max of normalized RGB values

    if (max_value == 0.0f) {
        *s = 0.0f;
        // Computes Saturation as 0 if max value is 0 to avoid division by zero
    } else {
        *s = (max_value - min_value) / max_value;
        // Otherwise, computes Saturation as the difference between max and min normalized RGB values divided by max value
    }

    if (max_value - min_value == 0.0f) {
        *h = 0.0f;
        // If all RGB channels are equal, Hue is set to 0 since the color is grayscale
    } else if (norm_r == max_value) {
        *h = ((norm_g - norm_b) / (max_value - min_value)) * 60.0f;
        if (*h < 0.0f) *h += 360.0f;
        // If Red is the max channel, computes Hue based on the difference between Green and Blue channels
    } else if (norm_g == max_value) {
        *h = ((norm_b - norm_r) / (max_value - min_value)) * 60.0f + 120.0f;
        // If Green is the max channel, computes Hue based on the difference between Blue and Red channels, offset by 120 degrees
    } else {
        *h = ((norm_r - norm_g) / (max_value - min_value)) * 60.0f + 240.0f;
        // If Blue is the max channel, computes Hue based on the difference between Red and Green channels, offset by 240 degrees
    }

    if (*h < 0.0f) *h += 360.0f;
    if (*h >= 360.0f) *h -= 360.0f;
    // Clamps hue into a valid range of [0, 360)
}

void rgbToHsvCPU(unsigned char *input_img, float *output_img, int height, int width) {
    // Convert ever pixel in the input image from RGB format to HSV 
    for (int y = 0; y < height; y++) {
        // Loop over each row of the image
        for (int x = 0; x < width; x++) {
            // Loop over each column of the image
            int pixel_index = (y * width + x) * 3;
            // Computes the starting index of the current pixel in the input image

            unsigned char r = input_img[pixel_index];
            unsigned char g = input_img[pixel_index + 1];
            unsigned char b = input_img[pixel_index + 2];
            // Extracts RGB components of the current pixel from the input image

            float h, s, v;
            // Temporary variables to hold the computed HSV values for the current pixel
            rgbToHsvSingle(r, g, b, &h, &s, &v);
            // Calls the helper function to convert the current pixel's RGB values to HSV

            output_img[pixel_index]     = h;
            output_img[pixel_index + 1] = s;
            output_img[pixel_index + 2] = v;
            // Stores the computed HSV values in the output image at the corresponding pixel index
        }
    }
}

void filterRedCPU(float *input_img, unsigned char *output_img, int height, int width) {
    // Detect Red pixels in the input HSV image and create a binary mask in the output image
    for (int y = 0; y < height; y++) {
        // Loop over each row of the image
        for (int x = 0; x < width; x++) {
            // Loop over each column of the image
            int pixel_index = (y * width + x) * 3;
            // Computes the starting index of the current pixel in the input HSV image

            float h = input_img[pixel_index];
            float s = input_img[pixel_index + 1];
            float v = input_img[pixel_index + 2];
            // Read HSV of current panel.

            int isRedHue = ((h >= 0.0f && h <= 10.0f) || (h >= 350.0f && h <= 360.0f));
            // Checks if hue corresponds to red color range 
            int isSaturated = (s > 0.3f);
            // Checks if saturation is enough to be a strong color
            int isBright = (v > 0.3f);
            // Checks if brightness is bright enough

            if (isRedHue && isSaturated && isBright) {
                // If all conditions met, mark as red
                output_img[y * width + x] = 255;
                // Sets red pixels as white
            } else {
                output_img[y * width + x] = 0;
                // Non-red pixels are set to black
            }
        }
    }
}

void gaussianSmoothCPU(unsigned char *input_img, unsigned char *output_img, int height, int width) {
    // Applies a Gaussian blur to the input binary mask to smooth out edges and reduce noise
    int half = GAUSSIAN_KERNEL_SIZE / 2;
    // Computes the half size of the Gaussian kernel for indexing

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Loops through every pixel
            float sum = 0.0f;
            // Accumulates the weighted sum of the neighboring pixels

            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    // Loops through 3x3 neighborhood around cur pixel
                    int neighbor_x = clampInt(x + kx, 0, width - 1);
                    int neighbor_y = clampInt(y + ky, 0, height - 1);
                    // Clamps the coordinates of neighbors so they remain in image boundaries

                    int neighbor_index = neighbor_y * width + neighbor_x;
                    // Compute the index of neighboring pixerls in grayscale image
                    float weight = gaussianKernel[ky + half][kx + half];
                    // Retrieves the gaussian weight corresponding to the neighbor pixel's position

                    sum += input_img[neighbor_index] * weight;
                    // Add the weighted neighbor to pixel value
                }
            }

            output_img[y * width + x] = (unsigned char)(sum / GAUSSIAN_NORM);
            // Normalizes the sum by the Gaussian normalization factor and stores the result in the output image
        }
    }
}