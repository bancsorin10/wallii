#include "wallii.h"


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

// save the model in a binary file under the following structure
// input_size1: unsigned int
// nr_weights1: unsigned int
// biases1: double [nr_weighst1]
// weights1: double [nr_weights1][input_size1]
// input_size2: unsigned int
// nr_weights2: unsigned int
// biases2: double [nr_weighst2]
// weights2: double [nr_weights2][input_size2]
static void save_model(t_sample_input *sample_in)
{
    int fd;
    unsigned int i;

    fd = open("wallii.model", O_CREAT|O_RDWR);
    write(fd, &sample_in->layer1->input_size, sizeof(unsigned int));
    write(fd, &sample_in->layer1->nr_weights, sizeof(unsigned int));
    write(fd, sample_in->layer1->biases, sizeof(double)*sample_in->layer1->nr_weights);
    for (i = 0; i < sample_in->layer1->nr_weights; ++i)
    {
        write(fd, sample_in->layer1->weights[i], sizeof(double)*sample_in->layer1->input_size);
    }
    write(fd, &sample_in->layer2->input_size, sizeof(unsigned int));
    write(fd, &sample_in->layer2->nr_weights, sizeof(unsigned int));
    write(fd, sample_in->layer2->biases, sizeof(double)*sample_in->layer2->nr_weights);
    for (i = 0; i < sample_in->layer2->nr_weights; ++i)
    {
        write(fd, sample_in->layer2->weights[i], sizeof(double)*sample_in->layer2->input_size);
    }
    close(fd);
}

static void avg_correction(t_correction **cor, t_correction *avg_cor, t_sample_input *sample)
{
    unsigned int s;
    unsigned int i;
    unsigned int j;

    double weights_sum;
    double bias_sum;
    for (i = 0; i < sample->layer1->nr_weights; ++i)
    {
        for (j = 0; j < sample->layer1->input_size; ++j)
        {
            weights_sum = 0;
            for (s = 0; s < BATCH_SIZE; ++s)
            {
                weights_sum += cor[s]->dweights1[i][j];
            }
            avg_cor->dweights1[i][j] = weights_sum/BATCH_SIZE;
        }

        bias_sum = 0;
        for (s = 0; s < BATCH_SIZE; ++s)
        {
            bias_sum += cor[s]->dbiases1[i];
        }
        avg_cor->dbiases1[i] = bias_sum/BATCH_SIZE;
    }
    for (i = 0; i < sample->layer2->nr_weights; ++i)
    {
        for (j = 0; j < sample->layer2->input_size; ++j)
        {
            weights_sum = 0;
            for (s = 0; s < BATCH_SIZE; ++s)
            {
                weights_sum += cor[s]->dweights2[i][j];
            }
            avg_cor->dweights2[i][j] = weights_sum/BATCH_SIZE;
        }

        bias_sum = 0;
        for (s = 0; s < BATCH_SIZE; ++s)
        {
            bias_sum += cor[s]->dbiases2[i];
        }
        avg_cor->dbiases2[i] = bias_sum/BATCH_SIZE;
    }
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
    output2_copy =
        (double *)malloc(sizeof(double)*sample_in->layer2->nr_weights);

    // forward propagation
    output1 = add_layer(in, sample_in->layer1);
    relu_activate(output1, sample_in->layer1->nr_weights);
    output2 = add_layer(output1, sample_in->layer2);
    output2_copy = memcpy(
            output2_copy,
            output2,
            sizeof(double)*sample_in->layer2->nr_weights);
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
                cor->dbiases1[j] +=
                    output2_copy[i]*sample_in->layer2->weights[i][j];
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
    unsigned int s;
    unsigned int epochs;
    double avg_loss;
    double avg_acc;
    double avg_epoch_loss;
    double avg_epoch_acc;
    double start_learn_rate;
    double curr_learn_rate;
    double learn_rate_decay;
    pthread_t *ptid;
    t_correction *avg_cor;
    t_sample_input *momentum;

    
    avg_cor = construct_correction(sample_in);
    momentum = construct_momentum(sample_in);
    ptid = (pthread_t *)malloc(sizeof(pthread_t)*BATCH_SIZE);
    sample_in->filename = input_files->files[0];
    start_learn_rate = CORR_COEFF;
    /* learn_rate_decay = 1e-6; */
    learn_rate_decay = 0;
    for (epochs = 0; epochs < NR_EPOCHS; ++epochs)
    {
        curr_learn_rate = start_learn_rate / (1/(1+learn_rate_decay*epochs));
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

        /*
         * The correction should be used from all the images, below only the
         * correction from the cor[0] is used, this should be averaged and a
         * momentum used in order to have better training.
         */
        avg_correction(cor, avg_cor, sample_in);
        /* for (s = 0; s < BATCH_SIZE; ++s) */
        /* { */
            /* for (i = 0; i < sample_in->layer1->nr_weights; ++i) */
            /* { */
                /* for (j = 0; j < sample_in->layer1->input_size; ++j) */
                /* { */
                /* } */
            /* } */
        /* } */

        // update weights
        for (i = 0; i < sample_in->layer1->nr_weights; ++i)
        {
            for (j = 0; j < sample_in->layer1->input_size; ++j)
            {
                sample_in->layer1->weights[i][j] -=
                    curr_learn_rate*avg_cor->dweights1[i][j];
            }
            sample_in->layer1->biases[i] -= curr_learn_rate*avg_cor->dbiases1[i];
        }
        for (i = 0; i < sample_in->layer2->nr_weights; ++i)
        {
            for (j = 0; j < sample_in->layer2->input_size; ++j)
            {
                sample_in->layer1->weights[i][j] -=
                    curr_learn_rate*avg_cor->dweights2[i][j];
            }
            sample_in->layer2->biases[i] -= curr_learn_rate*avg_cor->dbiases2[i];
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

    save_model(sample_in);

    free_correction(avg_cor);
    free_sample_input(sample_in);
    free_sample_input(momentum);

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
