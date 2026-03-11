#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "kernels.h"
extern "C" {
    #include "stb_image.h"
    #include "stb_image_write.h"
    #include "timing.h"
}

int main(int argc, char** argv ) {

    if ( argc != 2 )
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

    then = currentTime();

    unsigned char *device_rgb_pixels;
    float *device_hsv_pixels;
    int charByteSize = width * height * sizeof(unsigned char) * 3; // Each pixel has 3 components (R, G, B)
    int floatByteSize = width * height * sizeof(float) * 3; // Each pixel has 3 components (H, S, V)

    // Allocate device memory
    cudaMalloc(&device_rgb_pixels, charByteSize); 
    cudaMalloc(&device_hsv_pixels, floatByteSize); // Each pixel has 3 components (H, S, V)

    // Copy the RGB pixel data from host to device
    cudaMemcpy(device_rgb_pixels, img, charByteSize, cudaMemcpyHostToDevice);
    
    //Calculate Parameters
    dim3 block(16, 16); // Define block size
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    //Launch Kernel
    rgbToHsvKernel<<<grid, block>>>(device_rgb_pixels, device_hsv_pixels, width, height);

    cudaDeviceSynchronize();

    //Launch filterRed kernel to filter out the red pixels in the image
    filterRed<<<grid, block>>>(device_hsv_pixels, device_rgb_pixels, width, height);

    cudaDeviceSynchronize();

    // Copy result back to host and free device memory
    float *red_pixels = (float*)malloc(floatByteSize);
    cudaMemcpy(red_pixels, device_rgb_pixels, floatByteSize, cudaMemcpyDeviceToHost);

    now = currentTime();
    scost = now - then;

    stbi_write_png("output.png", width, height, 3, red_pixels, width * 3 * sizeof(float));
    printf("Image written to output.png\n");
    printf("Time taken for filtering red pixels: %f seconds\n", scost);

    cudaFree(device_hsv_pixels);
    cudaFree(device_rgb_pixels);
    free(red_pixels);
    free(img);

    return 0;
    
}