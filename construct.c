#include "wallii.h"

t_inputs *construct_inputs()
{
    t_inputs *input_files;
    struct dirent **namelist;
    unsigned int i;
    int n;
    char prefix[256];

    input_files = (t_inputs *)malloc(sizeof(t_inputs));
    n = scandir("inputs/", &namelist, NULL, alphasort);
    if (n < 1)
        exit(-2);

    n = (unsigned int)n;
    // ignore the `.` and `..` files
    input_files->nr_files = n-2;
    input_files->files = (char **)malloc(sizeof(char *)*(n-1));
    i = 2;
    while (i < n)
    {
        strcpy(prefix, "inputs/");
        input_files->files[i-2] = strdup(strcat(prefix, namelist[i]->d_name));
        ++i;
    }

    while (n--) {
        free(namelist[n]);
    }
    free(namelist);

    return input_files;
}

t_correction *construct_correction(t_sample_input *sample)
{
    unsigned int i;
    t_correction *cor;

    cor = (t_correction *)malloc(sizeof(t_correction));
    cor = (t_correction *)malloc(sizeof(t_correction));
    cor->dbiases1 = (double *)malloc(sizeof(double)*NR_WEIGHTS);
    cor->dbiases2 = (double *)malloc(sizeof(double)*NR_CLASSES);
    bzero(cor->dbiases1, sizeof(double)*NR_WEIGHTS);
    bzero(cor->dbiases2, sizeof(double)*NR_CLASSES);
    cor->dweights1 = (double **)malloc(sizeof(double *)*NR_WEIGHTS);
    cor->dweights2 = (double **)malloc(sizeof(double *)*NR_CLASSES);
    for (i = 0; i < NR_WEIGHTS; ++i)
    {
        cor->dweights1[i] = (double *)malloc(sizeof(double)*sample->layer1->input_size);
        bzero(cor->dweights1[i], sizeof(double)*sample->layer1->input_size);
    }
    for (i = 0; i < NR_CLASSES; ++i)
    {
        cor->dweights2[i] = (double *)malloc(sizeof(double)*sample->layer2->input_size);
        bzero(cor->dweights2[i], sizeof(double)*sample->layer2->input_size);
    }

    return cor;
}

t_sample_input *construct_initial()
{
    unsigned int i;
    unsigned int j;
    t_layer *layer1;
    t_layer *layer2;

    layer1 = (t_layer *)malloc(sizeof(t_layer));
    layer2 = (t_layer *)malloc(sizeof(t_layer));

    layer1->biases = (double *)malloc(sizeof(double)*NR_WEIGHTS);
    layer2->biases = (double *)malloc(sizeof(double)*NR_CLASSES);
    bzero(layer1->biases, sizeof(double)*NR_WEIGHTS);
    bzero(layer2->biases, sizeof(double)*NR_CLASSES);
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
    t_sample_input *sample_in;
    sample_in = (t_sample_input *)malloc(sizeof(t_sample_input));
    sample_in->layer1 = layer1;
    sample_in->layer2 = layer2;

    return sample_in;
}
