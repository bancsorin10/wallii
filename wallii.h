#ifndef wallii_h
# define wallii_h

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <string.h>
# include <fcntl.h>

/* # define ROWS 108 */
/* # define COLS 192 */
# define ROWS 1
# define COLS 3

// jpeg flags
# define SOI  0xffd8 // start of image
# define S0F0 0xffc0 // start of frame
# define S0F2 0xffc2 // start of frame
# define DHT  0xffc4 // define huffman tables
# define DQT  0xffdb // define quantization tables
# define DRI  0xffdd // define restart interval
# define SOS  0xffda // start of scan
# define RST0 0xffd0 // restart
# define RST1 0xffd1
# define RST2 0xffd2
# define RST3 0xffd3
# define RST4 0xffd4
# define RST5 0xffd5
# define RST6 0xffd6
# define RST7 0xffd7
# define APP0 0xffe0 // application specific
# define APP1 0xffe1
# define APP2 0xffe2
# define APP3 0xffe3
# define APP4 0xffe4
# define APP5 0xffe5
# define APP6 0xffe6
# define APP7 0xffe7
# define APP8 0xffe8
# define APP9 0xffe9
# define APPa 0xffea
# define APPb 0xffeb
# define APPc 0xffec
# define APPd 0xffed
# define APPe 0xffee
# define APPf 0xffef
# define COM  0xfffe // comment
# define EOI  0xffd9 // end of image

// each neuron should have the inputs, weights and a bias
// the pictures should be standard to ROWS|COLS
// we should start with 3 neurons one for each color (RGB)
// each pixel should have a weight assigned
typedef struct s_neuron {
    double pic[ROWS][COLS];
    double weight[ROWS][COLS];
    double bias;
}               t_neuron;

// expect the image to be saved as arrays for length `size` corresponding to
// the RGB values of the image
// the lengths could be dynamic but for the most part they will probably be
// of length 3888 - 1296x3 - (48x27x3)
typedef struct s_image {
    unsigned char *RGB;
    unsigned int size;
}               t_image;


#endif
