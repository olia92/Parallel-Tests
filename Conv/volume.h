#ifndef VOLUME_H
#define VOLUME_H

#include <inttypes.h>
#include <stddef.h>

// Volumes are used to represent the activations (i.e., state) between the
// different layers of the CNN. They all have three dimensions. The inter-
// pretation of their content depends on the layer that produced them. Before
// the first iteration, the Volume holds the data of the image we want to
// classify (the depth are the three color dimensions). After the last stage
// of the CNN, the Volume holds the probabilities that an image is part of
// a specific category.
//
// The weights are represented as a 1-d array with length
// width * height * depth.
typedef struct volume {
    int width;
    int height;
    int depth;
    double *weights;
} volume_t;

// Gets the element in the volume at the coordinates (x, y, d).
// #pragma acc routine seq
inline double volume_get(volume_t *v, int x, int y, int d);

// Sets the element in the volume at the coordinates (x, y, d) to value
// #pragma acc routine seq
inline void volume_set(volume_t *v, int x, int y, int d, double value);

//TEST:
void fdump_volume(volume_t* v,const char *file_name);

// Allocates a new volume with the specified dimensions, initializes it to the
// specified value.
volume_t *make_volume(int width, int height, int depth, double value);

// Copies the contents of one volume into another.
void copy_volume(volume_t *dest, volume_t *src);

// Frees the weights array and the struct itself.
void free_volume(volume_t *v);

//TEST:
volume_t *change_volume(volume_t *new_vol, double value);
volume_t *change_volume_acc(volume_t *new_vol, double value);
void copy_volume_host(volume_t *dest, volume_t *src);

// -------------------------------------------------------



// Convolutional Layer Parameters
typedef struct conv_layer {
    // Required
    int input_depth;
    int input_width;
    int input_height;
    int filter_width;
    int filter_height;
    int stride;
    int pad;
    int output_depth;

    // Computed
    int output_width;
    int output_height;
    double bias;
    volume_t *biases;
    volume_t **filters;
} conv_layer_t;

// Creates a convolutional layer with the following parameters.
conv_layer_t *make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
        int stride, int pad);

// Computes the forward pass for a convolutional layer on the relevant inputs
// and stores the result into the relevant outputs.
void conv_forward(conv_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end);

// Loads the convolutional layer weights from a file.
void conv_load(conv_layer_t *l, const char *file_name);


typedef struct network {
    volume_t *layers[2];
    conv_layer_t *l0;
    
} network_t;

// Creates a new instance of our network
network_t* make_network();

// Frees our network
void free_network(network_t* net);

typedef volume_t** batch_t;

network_t *make_network();

#endif
