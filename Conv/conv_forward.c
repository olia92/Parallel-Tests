// Performs the forward pass for a convolutional layer by convolving each one
// of the filters with a particular input, and placing the result in the output
// array.
//
// One way to think about convolution in this case is that we have one of the
// layer's filters (a 3D array) that is superimposed on one of the layer's
// inputs (a second 3D array) that has been implicitly padded with zeros. Since
// convolution is a sum of products (described below), we don't actually have
// to add any zeros to the input volume since those terms will not contribute
// to the convolution. Instead, for each position in the filter, we just make
// sure that we are in bounds for the input volume.
//
// Essentially, the filter is "sliding" across the input, in both the x and y
// directions, where we increment our position in each direction by using the
// stride parameter.
//
// At each position, we compute the sum of the elementwise product of the filter
// and the part of the array it's covering. For instance, let's consider a 2D
// case, where the filter (on the left) is superimposed on some part of the
// input (on the right).
//
//   Filter             Input
//  -1  0  1           1  2  3
//  -1  0  1           4  5  6
//  -1  0  1           7  8  9
//
// Here, the sum of the elementwise product is:
//    Filter[0][0] * Input[0][0] + Filter[0][1] * Input[0][1] + ...
//    = -1 * 1 + 0 * 2 + ... + 0 * 8 + 1 * 9
//    = 6
//
// The 3D case is essentially the same, we just have to sum over the other
// dimension as well. Also, since volumes are internally represented as 1D
// arrays, we must use the volume_get and volume_set commands to access elements
// at a coordinate (x, y, d). Finally, we add the corresponding bias for the
// filter to the sum before putting it into the output volume.

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <openacc.h>

#include "volume.h"

const char *DATA_FOLDER = "../../cifar-10-batches-bin";


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

conv_layer_t *make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
        int stride, int pad) {
    conv_layer_t *l = (conv_layer_t *) malloc(sizeof(conv_layer_t));
#pragma acc enter data create(l[0:1])
    l->output_depth = num_filters;
    l->filter_width = filter_width;
    l->input_depth = input_depth;
    l->input_width = input_width;
    l->input_height = input_height;

    l->filter_height = l->filter_width;
    l->stride = stride;
    l->pad = pad;

    l->output_width = (l->input_width + l->pad * 2 - l->filter_width) /
        l->stride + 1;
    l->output_height = (l->input_height + l->pad * 2 - l->filter_height) /
        l->stride + 1;
#pragma acc update device(l->output_depth,l->filter_width,l->input_depth,l->input_width,l->input_height,l->filter_height,l->stride,l->pad,l->output_width,l->output_height)
// #pragma acc update device(l[0:1])

    l->filters = malloc(sizeof(volume_t *) * num_filters);
#pragma acc enter data create(l->filters[0:num_filters])
    for (int i = 0; i < num_filters; i++) {
        l->filters[i] = make_volume(l->filter_width, l->filter_height,
            l->input_depth, 0.0);
    }

    l->bias = 0.0;
#pragma acc update device(l->bias)// xreiazetai
    l->biases = make_volume(1, 1, l->output_depth, l->bias);

    return l;
}

void conv_load(conv_layer_t *l, const char *file_name) {
    int filter_width, filter_height, depth, filters;

    FILE *fin = fopen(file_name, "r");

    fscanf(fin, "%d %d %d %d", &filter_width, &filter_height, &depth, &filters);
    assert(filter_width == l->filter_width);
    assert(filter_height == l->filter_height);
    assert(depth == l->input_depth);
    assert(filters == l->output_depth);

    for(int f = 0; f < filters; f++) {
        for (int x = 0; x < filter_width; x++) {
            for (int y = 0; y < filter_height; y++) {
                for (int d = 0; d < depth; d++) {
                    double val;
                    fscanf(fin, "%lf", &val);
                    volume_set(l->filters[f], x, y, d, val);
                }
            }
        }
    }

    for(int d = 0; d < l->output_depth; d++) {
        double val;
        fscanf(fin, "%lf", &val);
        volume_set(l->biases, 0, 0, d, val);
    }
//Update Weights and Biases on Device
    int we=filter_width*filter_height*depth;
    for(int f = 0; f < filters; f++) {
#pragma acc update device(l->filters[f]->weights[0:we])
    }

#pragma acc update device(l->biases->weights[0:l->output_depth])

    fclose(fin);
}

void conv_forward(conv_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
// #pragma acc parallel loop present(inputs,outputs,l,l->filters[0:l->output_depth])
    for (int i = start; i <= end; i++) {
        volume_t *in = inputs[i];
        volume_t *out = outputs[i];

        int stride = l->stride;

        for(int f = 0; f < l->output_depth; f++) {
            volume_t *filter = l->filters[f];
            int y = -l->pad;
            for(int out_y = 0; out_y < l->output_height; y += stride, out_y++) {
                int x = -l->pad;
                for(int out_x = 0; out_x < l->output_width; x += stride, out_x++) {

                    // Take sum of element-wise product
                    double sum = 0.0;
                    for(int fy = 0; fy < filter->height; fy++) {
                        int in_y = y + fy;
                        for(int fx = 0; fx < filter->width; fx++) {
                            int in_x = x + fx;
                            if(in_y >= 0 && in_y < in->height && in_x >=0 && in_x < in->width) {
                                for(int fd = 0; fd < filter->depth; fd++) {
                                    sum += volume_get(filter, fx, fy, fd) * volume_get(in, in_x, in_y, fd);
                                }
                            }
                        }
                    }

                    sum += l->biases->weights[f];
                    volume_set(out, out_x, out_y, f, sum);
                }
            }
        }
    }
}

// Function to dump the content of a volume for comparison.
void dump_volume(volume_t* v) {
    printf("%d,%d,%d", v->width, v->height, v->depth);
    for (int x = 0; x < v->width; x++) {
        for (int y = 0; y < v->height; y++) {
            for (int z = 0; z < v->depth; z++) {
                printf(",%.20lf", volume_get(v, x, y, z));
            }
        }
    }
    printf("\n");
}
typedef volume_t** batch_t;

// Load an entire batch of images from the cifar10 data set (which is divided
// into 5 batches with 10,000 images each).
batch_t load_batch(int batch) {
    printf("Loading input batch %d...\n", batch);

    char file_name[1024];
    sprintf(file_name, "%s/data_batch_%d.bin", DATA_FOLDER, batch+1);

    FILE *fin = fopen(file_name, "rb");
    assert(fin != NULL);
    batch_t batchdata = malloc(sizeof(volume_t *) * 10000);

    for (int i = 0; i < 10000; i++) {
        batchdata[i] = make_volume(32, 32, 3, 0.0);

        uint8_t data[3073];
        assert(fread(data, 1, 3073, fin) == 3073);

        int outp = 1;
        for (int d = 0; d < 3; d++) {
            for (int y = 0; y < 32; y++) {
                for (int x = 0; x < 32; x++) {
                    volume_set(batchdata[i], x, y, d, ((double)data[outp++])/255.0-0.5);
                }
            }
        }
    }

    fclose(fin);

    return batchdata;
}


int main(){

    printf("Convolution Layer Forward\n");

    conv_layer_t *l;
    volume_t **inputs; 
    volume_t **outputs;
    int start=0, end=1;
    int n = 10000;

    l = make_conv_layer(32, 32, 3, 5, 16, 1, 2);
    conv_load(l,"../../cs61c/snapshot/layer1_conv.txt");

    inputs = (volume_t **) malloc(sizeof(volume_t*)*n);
   
    inputs=load_batch(0);
    dump_volume(inputs[n-1]);
    outputs = (volume_t **) malloc(sizeof(volume_t*)*n);
    for(int i=0;i<n;i++){
        outputs[i]=make_volume(l->output_width,l->input_height,l->input_depth,0.0);
    }
   
    conv_forward(l,inputs, outputs,start, end);
    
    dump_volume(outputs[0]);
    

    return 0;

}