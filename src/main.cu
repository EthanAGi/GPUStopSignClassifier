#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "kernels.h"
#include "cpu_kernels.h"

extern "C" {
    #include "stb_image.h"
    #include "stb_image_write.h"
    #include "timing.h"
}

int main(int argc, char** argv ) {

    if (argc != 2)
    {
        printf("usage: ./project <Image_Path>\n");
        return -1;
    }

    double now, then;
    double scost, pcost;

    char *input_image_path = argv[1];

    printf("Reading image %s\n", input_image_path);

    // Load the image using stb_image
    int width, height, channels;
    unsigned char *img = stbi_load(input_image_path, &width, &height, &channels, 3);

    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    printf("Width: %d Height: %d\n", width, height);

    // ---------------- CPU PIPELINE ----------------
    float *cpu_hsv_pixels = (float*)malloc(width * height * 3 * sizeof(float));
    unsigned char *cpu_red_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *cpu_smoothed_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    if (!cpu_hsv_pixels || !cpu_red_pixels || !cpu_smoothed_pixels) {
        printf("Failed to allocate CPU memory\n");
        free(img);
        return 1;
    }

    then = currentTime();

    rgbToHsvCPU(img, cpu_hsv_pixels, height, width);
    filterRedCPU(cpu_hsv_pixels, cpu_red_pixels, height, width);
    gaussianSmoothCPU(cpu_red_pixels, cpu_smoothed_pixels, height, width);

    now = currentTime();
    pcost = (now - then) * 1000.0;

    stbi_write_png("cpu_smoothed_output.png", width, height, 1, cpu_smoothed_pixels, width);
    printf("Image written to cpu_smoothed_output.png\n");

    // ---------------- GPU PIPELINE ----------------
    then = currentTime();

    unsigned char *device_rgb_pixels;
    float *device_hsv_pixels;
    unsigned char *device_output_pixels;
    unsigned char *device_smoothed_pixels;

    int charByteSize = width * height * sizeof(unsigned char) * 3;
    int floatByteSize = width * height * sizeof(float) * 3;

    cudaMalloc(&device_rgb_pixels, charByteSize);
    cudaMalloc(&device_hsv_pixels, floatByteSize);
    cudaMalloc(&device_output_pixels, width * height * sizeof(unsigned char));
    cudaMalloc(&device_smoothed_pixels, width * height * sizeof(unsigned char));

    cudaMemcpy(device_rgb_pixels, img, charByteSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    rgbToHsvKernel<<<grid, block>>>(device_rgb_pixels, device_hsv_pixels, height, width);
    cudaDeviceSynchronize();

    filterRed<<<grid, block>>>(device_hsv_pixels, device_output_pixels, height, width);
    cudaDeviceSynchronize();

    gaussianSmoothKernel<<<grid, block>>>(device_output_pixels, device_smoothed_pixels, height, width);
    cudaDeviceSynchronize();

    unsigned char *red_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *smoothed_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    cudaMemcpy(red_pixels, device_output_pixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(smoothed_pixels, device_smoothed_pixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    now = currentTime();
    scost = (now - then) * 1000.0;

    stbi_write_png("smoothed_image_output.png", width, height, 1, smoothed_pixels, width);
    printf("Image written to smoothed_image_output.png\n");

    printf("CPU Time taken: %f ms\n", pcost);
    printf("GPU Time taken: %f ms\n", scost);
    if (scost > 0.0) {
        printf("Speedup: %.2fx\n", pcost / scost);
    }

    cudaFree(device_hsv_pixels);
    cudaFree(device_rgb_pixels);
    cudaFree(device_output_pixels);
    cudaFree(device_smoothed_pixels);

    free(cpu_hsv_pixels);
    free(cpu_red_pixels);
    free(cpu_smoothed_pixels);
    free(red_pixels);
    free(smoothed_pixels);
    free(img);

    return 0;
}