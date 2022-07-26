#ifndef wallii_h
# define wallii_h

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <string.h>
# include <fcntl.h>
# include <math.h>

# ifndef M_PI
# define M_PI 3.14159265358979323846
# endif

# define IMG_SIZE 3888
# define NR_WEIGHTS 5
# define NR_CLASSES 2
# define BATCH_SIZE 1

// expect the image to be saved as arrays for length `size` corresponding to
// the RGB values of the image
// the lengths could be dynamic but for the most part they will probably be
// of length 3888 - 1296x3 - (48x27x3)
typedef struct s_image
{
    unsigned char *RGB;
}               t_image;

typedef struct s_layer
{
    double **weights;
    double *biases;
    unsigned int nr_weights;
    unsigned int input_size;
}               t_layer;

t_image *decode_image(char *filename);
double random_normal();

#endif
