#include "wallii.h"

// the output is computed as the sum of matrix elements and the bias
// the matrix elements are computed by multiplying the inputs and the weights
// matrices element wise
double output_sum(t_neuron *neuron) {
    double out;
    unsigned int i;
    unsigned int j;

    out = neuron->bias;
    for (i = 0; i < ROWS; ++i) {
        for (j = 0; j < COLS; ++j) {
            out += neuron->pic[i][j]*neuron->weight[i][j];
        }
    }

    return out;
}


int main() {

    t_neuron *neuron;
    double in[COLS] = {1.0, 2.0, 3.0};
    double wei[COLS] = {0.2, 0.8, -0.5};

    neuron = (t_neuron *)malloc(sizeof(t_neuron));
    neuron->bias = 2;

    memcpy(neuron->pic, in, sizeof(in));
    memcpy(neuron->weight, wei, sizeof(wei));
    double out = output_sum(neuron);
    printf("%f\n", out);
    return 0;
}
