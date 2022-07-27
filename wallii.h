#ifndef wallii_h
# define wallii_h

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <string.h>
# include <fcntl.h>
# include <math.h>
# include <pthread.h>
# include <sys/types.h>
# include <dirent.h>

# ifndef M_PI
# define M_PI 3.14159265358979323846
# endif

# define IMG_SIZE 3888
# define NR_WEIGHTS 100
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

typedef struct s_correction
{
    double **dweights1;
    double **dweights2;
    double *dbiases1;
    double *dbiases2;
    double loss;
    double acc;
    unsigned char class; // 0 or 1
}           t_correction;

typedef struct s_sample_input {
    unsigned int thread_id;
    char *filename;
    t_layer *layer1;
    t_layer *layer2;
    t_correction *cor;
}           t_sample_input;

typedef struct s_inputs {
    char **files;
    unsigned int nr_files;
}           t_inputs;

t_image *decode_image(char *filename);
double random_normal();

#endif
