# Topic Proposal

## What is our Project going to do:

Our project will be fed an image and will detect if a stop sign is present. If a stop sign is detected, the output will be “Found”, otherwise “Not Found”. It will search for the stop sign and will modify the image by highlighting the location of the stop sign. We will be running parallel image processing kernels on the GPU to implement this. A CPU sequential solution will also be included in the project to demonstrate the speed differences of utilizing parallel computing as opposed to sequential programming. The program will look for common stop sign features like red color and octagonal shapes to determine if the stop sign is present.

**Project Report:**
https://docs.google.com/document/d/1-66TOopt5iPI4L7EljMpx3-c8KsVTwz5-ei495qnTxU/edit?usp=sharing

---

## Functionalities Implementation:

We will first implement the function to upload an RGB image on the CPU. We will process an image to find a stop sign using the CPU implementation utilizing sequential code. We will use an implemented timing function to record the length of completion time. Next, we time and search for a stop sign using the GPU implementation using parallel computing. The steps taken will include copying the 2D image into the GPU’s global memory. Threads will be mapped to each pixel of the image using a 1D array to represent the image on the GPU device. After timing, solving, and processing the image, the results will be copied back to the CPU’s memory where the output will be displayed.

- Exploring libraries such as OpenCV to implement our solution for shape and color detection  
- Implement GPU implementation utilizing parallel computing that maps the 2d image to threads  
- Implement CPU implementation that uses sequential code to search for stop signs in the 2d image.  
- Import/Implement Timing.c and Timing.h files to record the completion times of the GPU implementation along with the CPU implementation  
- Makefile to compile the code easily to promote simple use  

---

## Results:

- Correctly detects stop sign(s) in test images with fast GPU processing  
- Produce output images with the highlighted stop signs  
- Prints Timing results to the terminal comparing the CPU vs GPU implementations  
- Display speedups from the GPU implementation compared to the CPU  
- Validation of the stop sign “Found” or “Not Found”  

---

- Writeup of the project that includes images and videos of the project  
- README containing specifications of how to run the program
