#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

void rgbToHsvCPU(unsigned char *input_img, float *output_img, int height, int width);
void filterRedCPU(float *input_img, unsigned char *output_img, int height, int width);
void gaussianSmoothCPU(unsigned char *input_img, unsigned char *output_img, int height, int width);
void drawCircleCPU(unsigned char *pixels, int numRows, int numCols, int centerRow, int centerCol, float radius);

#endif