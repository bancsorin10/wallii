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
    bzero(output, sizeof(double)*layer->nr_weights);

    for (i = 0; i < layer->nr_weights; ++i)
    {
        output[i] = output_sum(input, layer->weights[i], layer->input_size, layer->biases[i]);
    }

    return output;
}

static void free_layer(t_layer *layer)
{
    unsigned int i;

    for (i = 0; i < layer->nr_weights; ++i)
    {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer->biases);
    free(layer);
}

static void free_correction(t_correction *cor)
{
    unsigned int i;

    for (i = 0; i < NR_WEIGHTS; ++i)
    {
        free(cor->dweights1[i]);
    }
    for (i = 0; i < NR_CLASSES; ++i)
    {
        free(cor->dweights2[i]);
    }
    free(cor->dbiases1);
    free(cor->dbiases2);
    free(cor);
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
    t_correction *cor;
    sample_in = (t_sample_input *)sample_input;
    cor = construct_correction(sample_in);

    img = decode_image(sample_in->filename);
    in = (double *)malloc(sizeof(double)*IMG_SIZE);
    bzero(in, sizeof(double)*IMG_SIZE);
    for (i = 0; i < IMG_SIZE; ++i)
        in[i] = (double)img->RGB[i];

    double *output2_copy;
    output2_copy = (double *)malloc(sizeof(double)*sample_in->layer2->nr_weights);

    // forward propagation
    output1 = add_layer(in, sample_in->layer1);
    relu_activate(output1, sample_in->layer1->nr_weights);
    output2 = add_layer(output1, sample_in->layer2);
    output2_copy = memcpy(output2_copy, output2, sizeof(double)*sample_in->layer2->nr_weights);
    softmax_activate(output2, sample_in->layer2->nr_weights);

    loss_function(output2, sample_in->filename, cor);

    // backward propagation
    output2_copy[cor->class] -= 1;
    for (i = 0; i < sample_in->layer2->nr_weights; ++i)
    {
        for (j = 0; j < sample_in->layer2->input_size; ++j)
        {
            cor->dweights2[i][j] = output1[j]*output2_copy[i];
        }
        cor->dbiases2[i] = output2_copy[i];
    }
    for (j = 0; j < sample_in->layer1->nr_weights; ++j)
    {
        cor->dbiases1[j] = 0;
        for (i = 0; i < sample_in->layer2->nr_weights; ++i)
        {
            if (output1[j] == 0)
                cor->dbiases1[j] = 0;
            else
                cor->dbiases1[j] += output2_copy[i]*sample_in->layer2->weights[i][j];
        }
    }
    for (i = 0; i < sample_in->layer1->nr_weights; ++i)
    {
        for (j = 0; j < sample_in->layer1->input_size; ++j)
        {
            cor->dweights1[i][j] = cor->dbiases1[i]*in[j];
        }
    }
    

    free(img->RGB);
    free(img);
    free(in);
    free(output1);
    free(output2);
    free(output2_copy);

    return (void *)cor;
}

static void train(
        t_sample_input *sample_in,
        t_correction **cor,
        t_inputs *input_files)
{
    unsigned int i;
    unsigned int j;
    unsigned int k;
    unsigned int epochs;
    double avg_loss;
    double avg_acc;
    double avg_epoch_loss;
    double avg_epoch_acc;
    pthread_t *ptid;

    
    ptid = (pthread_t *)malloc(sizeof(pthread_t)*BATCH_SIZE);
    sample_in->filename = input_files->files[0];
    for (epochs = 0; epochs < NR_EPOCHS; ++epochs)
    {
        avg_epoch_loss = 0;
        avg_epoch_acc  = 0;
    for (k = 0; k+BATCH_SIZE-1 < input_files->nr_files; k += BATCH_SIZE)
    {
        for (i = 0; i < BATCH_SIZE; ++i)
        {
            sample_in->filename = input_files->files[k+i];
            pthread_create(&ptid[i], NULL, &sample, sample_in);
        }
        for (i = 0; i < BATCH_SIZE; ++i)
        {
            pthread_join(ptid[i], &cor[i]);
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
        avg_acc /= BATCH_SIZE;
        avg_epoch_loss += avg_loss;
        avg_epoch_acc += avg_acc;

        // update weights
        for (i = 0; i < sample_in->layer1->nr_weights; ++i)
        {
            for (j = 0; j < sample_in->layer1->input_size; ++j)
            {
                sample_in->layer1->weights[i][j] -= CORR_COEFF*cor[0]->dweights1[i][j];
            }
            sample_in->layer1->biases[i] -= CORR_COEFF*cor[0]->dbiases1[i];
        }
        for (i = 0; i < sample_in->layer2->nr_weights; ++i)
        {
            for (j = 0; j < sample_in->layer2->input_size; ++j)
            {
                sample_in->layer1->weights[i][j] -= CORR_COEFF*cor[0]->dweights2[i][j];
            }
            sample_in->layer2->biases[i] -= CORR_COEFF*cor[0]->dbiases2[i];
        }

        // free correction
        for (i = 0; i < BATCH_SIZE; ++i)
        {
            free_correction(cor[i]);
        }
    }
    avg_epoch_loss /= (input_files->nr_files/BATCH_SIZE);
    avg_epoch_acc /= (input_files->nr_files/BATCH_SIZE);
    printf("epoch: %5d || avg loss: %5.5f || avg acc: %5.5f\n", epochs, avg_epoch_loss, avg_epoch_acc);
    }
    free(ptid);
}

int main()
{
    unsigned int i;
    t_correction **cor;
    t_sample_input *sample_in;
    t_inputs *input_files;

    cor = (t_correction **)malloc(sizeof(t_correction *)*BATCH_SIZE);
    input_files = construct_inputs();
    sample_in = construct_initial();
    train(sample_in, cor, input_files);

    // free memory
    free_layer(sample_in->layer1);
    free_layer(sample_in->layer2);
    for (i = 0; i < input_files->nr_files; ++i)
    {
        free(input_files->files[i]);
    }
    free(input_files->files);
    free(input_files);
    free(cor);
    free(sample_in);

    return 0;
}
