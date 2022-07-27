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

static void loss_function(double *output, char *filename, t_correction *cor)
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

static void *sample(void *sample_input)
{
    t_image *img;
    double *in;
    unsigned int i;
    unsigned int j;
    double *output1;
    double *output2;
    t_sample_input *sample_in;
    sample_in = (t_sample_input *)sample_input;

    /* pthread_detach(pthread_self()); */

    img = decode_image(sample_in->filename);
    in = (double *)malloc(sizeof(double)*IMG_SIZE);
    for (i = 0; i < IMG_SIZE; ++i)
        in[i] = (double)img->RGB[i];

    // forward propagation
    output1 = add_layer(in, sample_in->layer1);
    relu_activate(output1, sample_in->layer1->nr_weights);
    output2 = add_layer(output1, sample_in->layer2);
    softmax_activate(output2, sample_in->layer2->nr_weights);

    loss_function(output2, sample_in->filename, sample_in->cor);
    /* printf("output1: %10.5f | output2: %10.5f\n", output2[0], output2[1]); */

    // backward propagation
    output2[sample_in->cor->class] -= 1;
    for (i = 0; i < sample_in->layer2->nr_weights; ++i)
    {
        for (j = 0; j < sample_in->layer2->input_size; ++j)
        {
            sample_in->cor->dweights2[i][j] = output1[j]*output2[i];
        }
        sample_in->cor->dbiases2[i] = output2[i];
    }
    for (j = 0; j < sample_in->layer1->nr_weights; ++j)
    {
        sample_in->cor->dbiases1[j] = 0;
        for (i = 0; i < sample_in->layer2->nr_weights; ++i)
        {
            if (output1[j] == 0)
                sample_in->cor->dbiases1[j] = 0;
            else
                sample_in->cor->dbiases1[j] += output2[i]*sample_in->layer2->weights[i][j];
        }
    }
    for (i = 0; i < sample_in->layer1->nr_weights; ++i)
    {
        for (j = 0; j < sample_in->layer1->input_size; ++j)
        {
            sample_in->cor->dweights1[i][j] = sample_in->cor->dbiases1[i]*in[j];
        }
    }
    

    free(img->RGB);
    free(img);
    free(in);
    free(output1);
    free(output2);

    return NULL;
    /* pthread_exit(NULL); */
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

    t_correction **cor;

    cor = (t_correction **)malloc(sizeof(t_correction *)*BATCH_SIZE);
    for (i = 0; i < BATCH_SIZE; ++i)
    {
        cor[i] = (t_correction *)malloc(sizeof(t_correction));
        cor[i]->dbiases1 = (double *)malloc(sizeof(double)*NR_WEIGHTS);
        cor[i]->dbiases2 = (double *)malloc(sizeof(double)*NR_CLASSES);
        cor[i]->dweights1 = (double **)malloc(sizeof(double *)*NR_WEIGHTS);
        cor[i]->dweights2 = (double **)malloc(sizeof(double *)*NR_CLASSES);
        for (j = 0; j < NR_WEIGHTS; ++j)
        {
            cor[i]->dweights1[j] = (double *)malloc(sizeof(double)*layer1->input_size);
        }
        for (j = 0; j < NR_CLASSES; ++j)
        {
            cor[i]->dweights2[j] = (double *)malloc(sizeof(double)*layer2->input_size);
        }
    }

    unsigned int k;
    double avg_loss;
    double avg_acc;
    t_sample_input *sample_in;
    pthread_t *ptid;

    ptid = (pthread_t *)malloc(sizeof(pthread_t)*BATCH_SIZE);

    sample_in = (t_sample_input *)malloc(sizeof(t_sample_input));
    sample_in->filename = "inputs/zxzgpw_bad.rgb";
    sample_in->layer1 = layer1;
    sample_in->layer2 = layer2;
    sample_in->cor = cor[0];
    // for epochs
    // for 0-BATCH_SIZE start threads
    // average loss over the threads when collecting
    // update the weights
    for (k = 0; k < 5; ++k)
    {
        for (i = 0; i < BATCH_SIZE; ++i)
        {
            // sample_in->filename = ...
            sample_in->thread_id = i;
            sample_in->cor = cor[i];
            pthread_create(&ptid[i], NULL, &sample, sample_in);
            sample(sample_in);
        }
        for (i = 0; i < BATCH_SIZE; ++i)
        {
            pthread_join(ptid[i], NULL);
        }

        // average loss over the threads
        avg_loss = 0;
        avg_acc  = 0;
        for (i = 0; i < BATCH_SIZE; ++i)
        {
            avg_loss += cor[i]->loss;
            avg_acc  += cor[i]->acc;
        }
        avg_loss /= BATCH_SIZE;
        printf("%d || loss: %5.5f || acc: %5.5f\n", k, avg_loss, avg_acc);

        // update weights
        for (i = 0; i < layer1->nr_weights; ++i)
        {
            for (j = 0; j < layer1->input_size; ++j)
            {
                layer1->weights[i][j] -= 0.0001*cor[0]->dweights1[i][j];
            }
            layer1->biases[i] -= 0.0001*cor[0]->dbiases1[i];
        }
        for (i = 0; i < layer2->nr_weights; ++i)
        {
            for (j = 0; j < layer2->input_size; ++j)
            {
                layer1->weights[i][j] -= 0.0001*cor[0]->dweights2[i][j];
            }
            layer2->biases[i] -= 0.0001*cor[0]->dbiases2[i];
        }
    }

    // free memory
    free_layer(layer1);
    free_layer(layer2);
    for (i = 0; i < BATCH_SIZE; ++i)
    {
        for (j = 0; j < NR_WEIGHTS; ++j)
        {
            free(cor[i]->dweights1[j]);
        }
        for (j = 0; j < NR_CLASSES; ++j)
        {
            free(cor[i]->dweights2[j]);
        }
        free(cor[i]->dbiases1);
        free(cor[i]->dbiases2);
        free(cor[i]);
    }
    free(cor);

    return 0;
}
