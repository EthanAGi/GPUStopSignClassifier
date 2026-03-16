#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "kernels.h"
#include "cpu_kernels.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

extern "C" {
    #include "stb_image.h"
    #include "stb_image_write.h"
    #include "timing.h"
}

/**
 * A function that detects octagons in an image and draws circles around them. It uses OpenCV to find contours and approximate them to polygons. 
 * If a polygon has 8 vertices, it is considered an octagon. The center and radius of the minimum enclosing circle of the contour are calculated, 
 * and a circle is drawn around the detected octagon. The function can run on either the CPU or GPU based on the 'gpu' boolean parameter. 
 * If 'gpu' is true, it will use the drawCircleKernel to draw circles on the GPU; otherwise, it will use a CPU implementation. 
 * The modified image with detected octagons will be saved as output.
 */
void detectAndDrawOctagon(cv::Mat &cpu_edges, cv::Mat &original_img, unsigned char *device_draw_pixels, int width, int height, dim3 grid, dim3 block, bool gpu) {
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cpu_edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    int octagon_count = 0;

    for (auto& contour : contours) {

        if (cv::contourArea(contour) < 1000) continue;

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.02 * cv::arcLength(contour, true), true);

        if (approx.size() == 8) {

            octagon_count++;

            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);

            int centerRow = (int)center.y;
            int centerCol = (int)center.x;
            radius += 20;

            printf("\nStop sign detected! Center: (%d, %d) Radius: %.1f\n",
                   centerCol, centerRow, radius);

            if (gpu) {

                drawCircleKernel<<<grid, block>>>(device_draw_pixels, height, width, centerRow, centerCol, radius);
                cudaDeviceSynchronize();

            } else {

                drawCircleCPU(original_img.data, height, width, centerRow, centerCol, radius);

            }

        }

    }

    if (gpu) {
        cudaMemcpy(original_img.data, device_draw_pixels, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    }

    if (octagon_count == 0) {
        printf("No stop sign detected.\n");
    }

}

int main(int argc, char** argv ) {

    if (argc != 2)
    {
        printf("usage: ./project <Image_Path>\n");
        return -1;
    }

    double now, then;
    double scost, pcost;

    double kernel_then;
    double gpu_kernel_time;

    char *input_image_path = argv[1];

    printf("Reading image %s\n", input_image_path);

    // Load the image using stb_image
    int width, height, channels;
    unsigned char *img = stbi_load(input_image_path, &width, &height, &channels, 3);

    if (!img) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return 1;
    }

    // Define block and grid sizes for CUDA kernels
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    printf("Width: %d Height: %d\n\n", width, height);

    // ------------------------------------------------------------------------------------------------
    // ----------------------------------------- CPU PIPELINE -----------------------------------------
    // ------------------------------------------------------------------------------------------------

    // Allocate memory for intermediate results on the CPU
    float *cpu_hsv_pixels = (float*)malloc(width * height * 3 * sizeof(float));
    unsigned char *cpu_red_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *cpu_smoothed_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    if (!cpu_hsv_pixels || !cpu_red_pixels || !cpu_smoothed_pixels) {
        printf("Failed to allocate CPU memory\n");
        free(img);
        return 1;
    }

    then = currentTime();

    //Run the CPU methods for RGB to HSV conversion, red color filtering, and Gaussian smoothing
    rgbToHsvCPU(img, cpu_hsv_pixels, height, width);
    filterRedCPU(cpu_hsv_pixels, cpu_red_pixels, height, width);
    gaussianSmoothCPU(cpu_red_pixels, cpu_smoothed_pixels, height, width);

    //Use OpenCV library to run Canny Edge Detection
    cv::Mat cpu_smoothed_mat(height, width, CV_8UC1, cpu_smoothed_pixels);

    cv::Mat cpu_canny_edges;
    cv::Canny(cpu_smoothed_mat, cpu_canny_edges, 50.0, 150.0);
    
    // ----------------------------------------- CPU OCTAGON DETECTION -----------------------------------------

    //Use OpenCV library to run contour detection and octagon approximation
    cv::Mat cpu_original_rgb(height, width, CV_8UC3, img);
    cv::Mat cpu_original;
    cv::cvtColor(cpu_original_rgb, cpu_original, cv::COLOR_RGB2BGR);

    printf("\n-------------------------CPU STOP SIGN DETECTION-------------------------\n");

    detectAndDrawOctagon(cpu_canny_edges, cpu_original, nullptr, width, height, grid, block, false);

    now = currentTime();
    scost = (now - then) * 1000.0;

    stbi_write_png("cpu_smoothed_output.png", width, height, 1, cpu_smoothed_pixels, width);
    cv::imwrite("cpu_canny_edges_output.png", cpu_canny_edges);
    cv::imwrite("cpu_detected_output.png", cpu_original);
    printf("Detection result written to cpu_detected_output.png\n\n");
    printf("Images written to cpu_smoothed_output.png and cpu_canny_edges_output.png\n");

    // ------------------------------------------------------------------------------------------------
    // ----------------------------------------- GPU PIPELINE -----------------------------------------
    // ------------------------------------------------------------------------------------------------

    then = currentTime();

    // Allocate memory for intermediate results on the GPU
    unsigned char *device_rgb_pixels;
    float *device_hsv_pixels;
    unsigned char *device_output_pixels;
    unsigned char *device_smoothed_pixels;

    // Calculate byte sizes for memory allocation
    int charByteSize = width * height * sizeof(unsigned char) * 3;
    int floatByteSize = width * height * sizeof(float) * 3;

    //Allocate memory on the GPU for the RGB pixels, HSV pixels, output pixels after red filtering, and smoothed pixels after Gaussian smoothing
    cudaMalloc(&device_rgb_pixels, charByteSize);
    cudaMalloc(&device_hsv_pixels, floatByteSize);
    cudaMalloc(&device_output_pixels, width * height * sizeof(unsigned char));
    cudaMalloc(&device_smoothed_pixels, width * height * sizeof(unsigned char));

    //Copy the RGB pixel data from the host (CPU) to the device (GPU)
    cudaMemcpy(device_rgb_pixels, img, charByteSize, cudaMemcpyHostToDevice);

    //Timing for only kernel execution
    kernel_then = currentTime();

    //Run the GPU kernels for RGB to HSV conversion, red color filtering, and Gaussian smoothing
    rgbToHsvKernel<<<grid, block>>>(device_rgb_pixels, device_hsv_pixels, height, width);
    cudaDeviceSynchronize();

    filterRed<<<grid, block>>>(device_hsv_pixels, device_output_pixels, height, width);
    cudaDeviceSynchronize();

    gaussianSmoothKernel<<<grid, block>>>(device_output_pixels, device_smoothed_pixels, height, width);
    cudaDeviceSynchronize();

    //Use OpenCV library to run Canny Edge Detection on the GPU
    cv::Mat cpu_smoothed_temp(height, width, CV_8UC1);
    cudaMemcpy(cpu_smoothed_temp.data, device_smoothed_pixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //Upload the smoothed image to the GPU
    cv::cuda::GpuMat gpu_smoothed;
    gpu_smoothed.upload(cpu_smoothed_temp);
    cv::cuda::GpuMat gpu_edges;

    //Run Canny Edge Detection on the GPU using OpenCV's CUDA module
    auto canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0);
    canny->detect(gpu_smoothed, gpu_edges);
    
    //Download edges and saves them
    cv::Mat cpu_edges;
    gpu_edges.download(cpu_edges);

    // ----------------------------------------- GPU OCTAGON DETECTION -----------------------------------------

    //Use OpenCV library to run contour detection and octagon approximation on the GPU
    cv::Mat original_img_rgb(height, width, CV_8UC3, img);
    cv::Mat original_img;
    cv::cvtColor(original_img_rgb, original_img, cv::COLOR_RGB2BGR);

    //Allocate memory on the GPU for drawing the detected octagons
    unsigned char *device_draw_pixels = nullptr;
    cudaMalloc((void**)&device_draw_pixels, width * height * 3 * sizeof(unsigned char));
    cudaMemcpy(device_draw_pixels, original_img.data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    printf("\n\n-------------------------GPU STOP SIGN DETECTION-------------------------\n");

    detectAndDrawOctagon(cpu_edges, original_img, device_draw_pixels, width, height, grid, block, true);

    now = currentTime();
    pcost = (now - then) * 1000.0;
    gpu_kernel_time = (now - kernel_then) * 1000.0;

    //Copy the modified image with detected octagons back to the host and save it
    cv::imwrite("gpu_detected_output.png", original_img);
    printf("Detection result written to gpu_detected_output.png\n\n");

    cudaFree(device_draw_pixels);

    //Copy the smoothed image and edges back to the host and save them
    unsigned char *smoothed_pixels = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    cudaMemcpy(smoothed_pixels, device_smoothed_pixels, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::imwrite("gpu_canny_edges_output.png", cpu_edges);
    stbi_write_png("gpu_smoothed_output.png", width, height, 1, smoothed_pixels, width);
    printf("Images written to gpu_smoothed_output.png and gpu_canny_edges_output.png\n\n");

    printf("\n-------------------------TIMING OUTPUTS-------------------------\n");
    printf("CPU Time taken: %f ms\n", scost);
    printf("GPU Time taken: %f ms\n", pcost);
    if (scost > 0.0) {
        printf("Speedup: %.3fx\n\n", scost / pcost);
    }

    printf("ONLY GPU Kernel execution time: %f ms\n", gpu_kernel_time);
    if (scost > 0.0) {
        printf("Kernel ONLY Speedup: %.2fx\n\n", scost / gpu_kernel_time);
    }

    cudaFree(device_hsv_pixels);
    cudaFree(device_rgb_pixels);
    cudaFree(device_output_pixels);
    cudaFree(device_smoothed_pixels);

    free(cpu_hsv_pixels);
    free(cpu_red_pixels);
    free(cpu_smoothed_pixels);
    free(smoothed_pixels);
    free(img);

    return 0;
}