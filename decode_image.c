#include "wallii.h"

// function to read the image and choose between png|jpeg decoding strategies
// return: decoded image in RGB format
// expect: files in `.rgb` aka raw format of a certain size
t_image *decode_image(char *filename)
{
    t_image *img;
    int fd;

    img = (t_image *)malloc(sizeof(t_image));
    fd = open(filename, O_RDONLY);

    // dummy initialization
    img->RGB = (unsigned char *)malloc(sizeof(unsigned char)*(IMG_SIZE + 1));
    read(fd, img->RGB, IMG_SIZE);
    img->RGB[IMG_SIZE] = '\0'; // extra

    close(fd);

    return img;
}
