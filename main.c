#include "wallii.h"

// compute the sum of elementwise multiplication of the inputs and the weights
// + the bias
static double output_sum(
        double *input,
        double *weight,
        unsigned int size,
        double bias)
{

    unsigned int i;
    double out;

    out = bias;
    for (i = 0; i < size; ++i)
    {
        out += input[i]*weight[i];
    }

    /* printf("suma:\n"); */
    /* for (i = 0; i < size; ++i) { */
            /* printf("input: %-15.5f | weight: %-15.5f | mul: %-15.5f\n", input[i], weight[i], input[i]*weight[i]); */
    /* } */
    /* printf("out: %.5f\n", out); */

    return out;
}

// compute a new layer
// should probably return a structure of the used weights and the results of
// `output_sum` for each input with each weight
static double *add_layer(
        double *input,
        t_layer *layer)
{
    unsigned int i;
    double *output;

    output = (double *)malloc(sizeof(double)*layer->nr_weights);

    for (i = 0; i < layer->nr_weights; ++i)
    {
        // compute the output
        output[i] = output_sum(input, layer->weights[i], layer->input_size, layer->biases[i]);
    }

    return output;
}

static void free_layer(t_layer *layer) {
    unsigned int i;

    for (i = 0; i < layer->nr_weights; ++i)
    {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    free(layer);
}

// relu activation function
static void relu_activate(double *output, unsigned int size)
{
    unsigned int i;

    /* printf("output1 before relu:\n"); */
    /* for (i = 0; i < 5; ++i) { */
        /* printf("%.10f ", output[i]); */
    /* } */
    /* printf("\n"); */

    for (i = 0; i < size; ++i)
    {
        output[i] = output[i]*(output[i] > 0);
    }
}

// softmax activation function
static void softmax_activate(double *output, unsigned int size)
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

static double loss_function(double *output, char *filename)
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

    return loss;
}

static void forward(
        unsigned int thread_id,
        char *filename,
        t_layer *layer1,
        t_layer *layer2,
        double *loss)
{
    t_image *img;
    double *in;
    unsigned int i;
    double *output1;
    double *output2;

    img = decode_image(filename);
    in = (double *)malloc(sizeof(double)*IMG_SIZE);
    for (i = 0; i < IMG_SIZE; ++i)
        in[i] = (double)img->RGB[i];

    output1 = add_layer(in, layer1);
    relu_activate(output1, layer1->nr_weights);
    output2 = add_layer(output1, layer2);
    softmax_activate(output2, layer2->nr_weights);

    loss[thread_id] = loss_function(output2, filename);
    printf("output1: %10.5f | output2: %10.5f\n", output2[0], output2[1]);
    free(img->RGB);
    free(img);
    free(in);
}

int main()
{
    unsigned int i;
    unsigned int j;
    t_layer *layer1;
    t_layer *layer2;

    layer1 = (t_layer *)malloc(sizeof(t_layer));
    layer2 = (t_layer *)malloc(sizeof(t_layer));

    layer1->biases = (double *)malloc(sizeof(double)*NR_WEIGHTS);
    layer2->biases = (double *)malloc(sizeof(double)*NR_WEIGHTS);
    layer1->input_size = IMG_SIZE;
    layer1->nr_weights = NR_WEIGHTS;
    layer2->input_size = layer1->nr_weights;
    layer2->nr_weights = NR_CLASSES;

    layer1->weights = (double **)malloc(sizeof(double *)*NR_WEIGHTS);
    srand(42); // set the seed
    for (i = 0; i < NR_WEIGHTS; ++i) {
        // allocate memory and randomize the weights
        layer1->weights[i] = (double *)malloc(sizeof(double)*IMG_SIZE);
        for (j = 0; j < layer1->input_size; ++j) {
            layer1->weights[i][j] = random_normal() * 0.01; // small weights
        }
    }


    layer2->weights = (double **)malloc(sizeof(double *)*NR_CLASSES);
    for (i = 0; i < NR_CLASSES; ++i) {
        layer2->weights[i] = (double *)malloc(sizeof(double)*layer2->input_size);
        for (j = 0; j < layer2->input_size; ++j) {
            layer2->weights[i][j] = random_normal() * 0.01; // small weights
        }
    }

    double *loss;
    double avg_loss;
    loss = (double *)malloc(sizeof(double)*BATCH_SIZE);

    // for epochs
    // for 0-BATCH_SIZE start threads
    // average loss over the threads when collecting
    // update the weights
    forward(0, "inputs/zxzgpw_bad.rgb", layer1, layer2, loss);
    printf("%.5f\n", loss[0]);

    // average loss over the threads
    avg_loss = 0;
    for (i = 0; i < BATCH_SIZE; ++i)
        avg_loss += loss[i];
    avg_loss /= BATCH_SIZE;

    // free memory
    free_layer(layer1);
    free_layer(layer2);

    return 0;
}
