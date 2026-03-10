#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

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

    char *input_image_path = argv[1];

    printf("Reading image %s\n", input_image_path);


    // Load the image using stb_image
    int width, height, channels;

    //Represented in a 1D array of unsigned chars in the format of [ R, G, B, R, G, B, R, G, B, ... ]
    unsigned char *img = stbi_load(input_image_path, &width, &height, &channels, 3);

    if (!img) {
        printf("Failed to load image\n");
        return 1;
    }

    printf("Width: %d Height: %d\n", width, height);

    /* Access RGB pixels
    
        int index = (y * width + x) * 3;
        unsigned char r = img[index + 0];
        unsigned char g = img[index + 1];
        unsigned char b = img[index + 2];
    
        This is the mapping required because the output is a 1D array of unsigned chars in the repeating order of {R, G, B}
    */
    for (int y = 0; y < height; y++) {

        for (int x = 0; x < width; x++) {

            int index = (y * width + x) * 3;

            unsigned char r = img[index + 0];
            unsigned char g = img[index + 1];
            unsigned char b = img[index + 2];

            printf("(%d,%d,%d) ", r, g, b);
        }
        printf("\n");

    }

    // Free the image memory
    free(img);

    return 0;
    
}