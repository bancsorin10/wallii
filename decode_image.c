#include "wallii.h"

// function to read the image and choose between png|jpeg decoding strategies
// return: decoded image in RGB format
// expect: files in `.rgb` aka raw format of a certain size
t_image *decode_image(char *filename)
{
    t_image *img;
    int fd;
    unsigned int i = 0;

    img = (t_image *)malloc(sizeof(t_image));
    fd = open(filename, O_RDONLY);

    // dummy initialization
    img->RGB = (unsigned char *)malloc(sizeof(unsigned char)*11);
    img->RGB = memset(img->RGB, 'a', 10);
    img->RGB[10] = '\0';
    img->size = 10;

    close(fd);

    return img;
}
