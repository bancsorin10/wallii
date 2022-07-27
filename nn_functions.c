#include "wallii.h"

// relu activation function
void relu_activate(double *output, unsigned int size)
{
    unsigned int i;

    for (i = 0; i < size; ++i)
    {
        output[i] = output[i]*(output[i] > 0);
    }
}

// softmax activation function
void softmax_activate(double *output, unsigned int size)
{
    unsigned int i;
    double sum = 0;
    double max = output[0];

    // get max element to substract from the outputs to prevent overflows when
    // exponentiating
    for (i = 0; i < size; ++i)
    {
        if (max < output[i])
            max = output[i];
    }

    // exponentiate
    for (i = 0; i < size; ++i)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    // normalize
    for (i = 0; i < size; ++i)
    {
        output[i] = output[i]/sum;
    }
}

void loss_function(double *output, char *filename, t_correction *cor)
{
    double loss;
    unsigned int i;
    unsigned char class;
    double out_clip;

    // assume the class is defined in the `filename` - god/bad
    i = 0;
    while (filename[i])
    {
        if (filename[i] == '_')
        {
            if (filename[i+1] == 'g')
                class = 0;
            else
                class = 1;
            break;
        }
        ++i;
    }

    if (output[class] > (1-1e-7))
        out_clip = 1-1e-7;
    else if (output[class] < 1e-7)
        out_clip = 1e-7;
    else
        out_clip = output[class];

    loss = -log(out_clip);

    cor->loss  = loss;
    cor->acc   = output[class] > 0.5;
    cor->class = class;
}
