
#include "wallii.h"

static t_sample_input *load_model(char *filename)
{
    t_sample_input *model;
    unsigned int i;
    int fd;

    model = (t_sample_input *)malloc(sizeof(t_sample_input));
    model->layer1 = (t_layer *)malloc(sizeof(t_layer));
    model->layer2 = (t_layer *)malloc(sizeof(t_layer));

    fd = open(filename, O_RDONLY);

    read(fd, &model->layer1->input_size, sizeof(unsigned int));
    read(fd, &model->layer1->nr_weights, sizeof(unsigned int));
    model->layer1->biases = (double *)malloc(sizeof(double)*model->layer1->nr_weights);
    read(fd, model->layer1->biases, sizeof(double)*model->layer1->nr_weights);
    model->layer1->weights = (double **)malloc(sizeof(double *)*model->layer1->nr_weights);
    for (i = 0; i < model->layer1->nr_weights; ++i)
    {
        model->layer1->weights[i] = (double *)malloc(sizeof(double)*model->layer1->input_size);
        read(fd, model->layer1->weights[i], sizeof(double)*model->layer1->input_size);
    }
    read(fd, &model->layer2->input_size, sizeof(unsigned int));
    read(fd, &model->layer2->nr_weights, sizeof(unsigned int));
    model->layer2->biases = (double *)malloc(sizeof(double)*model->layer2->nr_weights);
    read(fd, model->layer2->biases, sizeof(double)*model->layer2->nr_weights);
    model->layer2->weights = (double **)malloc(sizeof(double *)*model->layer2->nr_weights);
    for (i = 0; i < model->layer2->nr_weights; ++i)
    {
        model->layer2->weights[i] = (double *)malloc(sizeof(double)*model->layer2->input_size);
        read(fd, model->layer2->weights[i], sizeof(double)*model->layer2->input_size);
    }

    close(fd);
    
    return model;
}

static void validate(t_sample_input *model, char *filename)
{
    double *output1;
    double *output2;
    unsigned int i;
    char *file_in;
    double *in;
    int fd;

    file_in = (char *)malloc(sizeof(char)*IMG_SIZE);
    in = (double *)malloc(sizeof(double)*IMG_SIZE);
    fd = open(filename, O_RDONLY);
    read(fd, file_in, IMG_SIZE);
    for (i = 0; i < IMG_SIZE; ++i)
    {
        in[i] = (double)file_in[i];
    }
    free(file_in);
    close(fd);
    
    output1 = add_layer(in, model->layer1);
    relu_activate(output1, model->layer1->nr_weights);
    output2 = add_layer(output1, model->layer2);
    softmax_activate(output2, model->layer2->nr_weights);

    printf("%s --- good: %5.5f | bad: %5.5f\n", filename, output2[0], output2[1]);

}

int main(int ac, char **av)
{
    if (ac == 1)
        return (-2);

    t_sample_input *model;
    model = load_model("wallii.model");
    validate(model, av[1]);

    return 0;
}
