#include "wallii.h"

static void decode_jpeg(int fd, t_image *img) {
}

static void decode_png(int fd, t_image *img) {
}

// function to read the image and choose between png|jpeg decoding strategies
// return: decoded image in RGB format
t_image *decode_image(char *filename) {
    t_image *img;
    int fd;
    int i;

    img = (t_image *)malloc(sizeof(t_image));
    fd = open(filename, O_RDONLY);

    while (filename[i]) {
        i++;
        if (filename[i-1] != '.')
            continue;
        if (filename[i] == 'j')
            decode_jpeg(fd, img);
        else
            decode_png(fd, img);
        break;
    }

    close(fd);

    return img;
}
