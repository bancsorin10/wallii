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
static void add_layer(
        double *input,
        t_layer *layer)
{
    unsigned int i;
    if (!layer->output)
        layer->output = (double *)malloc(sizeof(double)*layer->nr_weights);

    for (i = 0; i < layer->nr_weights; ++i)
    {
        // compute the output
        layer->output[i] = output_sum(input, layer->weights[i], layer->input_size, layer->biases[i]);
    }
}

static void free_layer(t_layer *layer) {
    unsigned int i;

    for (i = 0; i < layer->nr_weights; ++i)
    {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    free(layer->output);
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

static double loss(double *output, char *filename)
{
    double loss;
    unsigned int i;
    unsigned char class;
    double out_clip;

    // assume the class is defined in the `filename` - god/bad
    i = 0;
    while (filename[i])
    {
        if (filename[i] == '-')
        {
            if (filename[i+1] == 'g')
                class = 1;
            else
                class = 0;
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
    t_layer *layer1_copy;
    t_layer *layer2_copy;
}

int main()
{
    unsigned int i;
    unsigned int j;
    unsigned int nr_weights = 100;
    unsigned int nr_classes = 2; // good or bad
    t_image *img;
    t_layer *layer1;
    t_layer *layer2;

    img = decode_image("images/011km1.jpg");

    layer1 = (t_layer *)malloc(sizeof(t_layer));
    layer2 = (t_layer *)malloc(sizeof(t_layer));

    layer1->biases = (double *)malloc(sizeof(double)*nr_weights);
    layer2->biases = (double *)malloc(sizeof(double)*nr_weights);
    layer1->input_size = img->size;
    layer1->nr_weights = nr_weights;
    layer2->input_size = layer1->nr_weights;
    layer2->nr_weights = nr_classes;

    layer1->weights = (double **)malloc(sizeof(double *)*nr_weights);
    srand(42); // set the seed
    for (i = 0; i < nr_weights; ++i) {
        // allocate memory and randomize the weights
        layer1->weights[i] = (double *)malloc(sizeof(double)*img->size);
        for (j = 0; j < layer1->input_size; ++j) {
            layer1->weights[i][j] = random_normal() * 0.01; // small weights
        }
    }

    // first layer inputs the actual image and generated weights for each
    // pixel
    double *iin = (double *) malloc (sizeof(double)*img->size);
    for (i = 0; i < img->size; ++i) {
        iin[i] = (double)img->RGB[i];
    }
    add_layer(iin, layer1);
    relu_activate(layer1->output, nr_weights);

    layer2->weights = (double **)malloc(sizeof(double *)*nr_classes);
    for (i = 0; i < nr_classes; ++i) {
        layer2->weights[i] = (double *)malloc(sizeof(double)*nr_weights);
        for (j = 0; j < layer2->input_size; ++j) {
            layer2->weights[i][j] = random_normal() * 0.01; // small weights
        }
    }

    add_layer(layer1->output, layer2);
    softmax_activate(layer2->output, layer2->nr_weights);

    /* printf("weights1:\n"); */
    /* for (i = 0; i < 10; ++i) { */
        /* printf("%.10f ", layer1->weights[0][i]); */
    /* } */
    /* printf("\n"); */
    /* printf("output1:\n"); */
    /* for (i = 0; i < 5; ++i) { */
        /* printf("%.10f ", layer1->output[i]); */
    /* } */
    /* printf("\n"); */

    printf("%.5f -- %.5f\n", layer2->output[0], layer2->output[1]);

    // free memory
    free_layer(layer1);
    free_layer(layer2);
    free(img->RGB);
    free(img);

    return 0;
}
