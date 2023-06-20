#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <openacc.h>

#include "volume.h"

// #pragma acc routine seq
inline double volume_get(volume_t *v, int x, int y, int d) {
    return v->weights[((v->width * y) + x) * v->depth + d];
}
// #pragma acc routine seq
inline void volume_set(volume_t *v, int x, int y, int d, double value) {
    v->weights[((v->width * y) + x) * v->depth + d] = value;
}

volume_t *make_volume(int width, int height, int depth, double value) {
    volume_t *new_vol = malloc(sizeof(struct volume));
    new_vol->weights = malloc(sizeof(double) * width * height * depth);

    new_vol->width = width;
    new_vol->height = height;
    new_vol->depth = depth;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int d = 0; d < depth; d++) {
                volume_set(new_vol, x, y, d, value);
            }
        }
    }

    return new_vol;
}

void copy_volume(volume_t *dest, volume_t *src) {
    assert(dest->width == src->width);
    assert(dest->height == src->height);
    assert(dest->depth == src->depth);

    for (int x = 0; x < dest->width; x++) {
        for (int y = 0; y < dest->height; y++) {
            for (int d = 0; d < dest->depth; d++) {
                volume_set(dest, x, y, d, volume_get(src, x, y, d));
            }
        }
    }
}

void free_volume(volume_t *v) {
    free(v->weights);
    free(v);
}

// Function to dump the content of a volume for comparison.
void dump_volume(volume_t* v) {
    printf("%d,%d,%d", v->width, v->height, v->depth);
    for (int z = 0; z < v->depth; z++) {
    for (int x = 0; x < v->width; x++) {
        for (int y = 0; y < v->height; y++) {
            
                printf(",%.3lf", volume_get(v, x, y, z));//.20
            }
        }
    }
    printf("\n");
}


//TEST: Volume to TXT
void fdump_volume(volume_t* v,const char *file_name) {

    FILE *fin = fopen(file_name, "w");

    fprintf(fin,"%d %d %d\n", v->width, v->height, v->depth);
    for (int x = 0; x < v->width; x++) {
        for (int y = 0; y < v->height; y++) {
            for (int z = 0; z < v->depth; z++) {
                fprintf(fin,"%.20lf\n", volume_get(v, x, y, z));
            }
        }
    }
    fclose(fin);
}


conv_layer_t *make_conv_layer(int input_width, int input_height, int input_depth, int filter_width, int num_filters,
        int stride, int pad) {
    conv_layer_t *l = (conv_layer_t *) malloc(sizeof(conv_layer_t));

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

    l->filters = malloc(sizeof(volume_t *) * num_filters);
    for (int i = 0; i < num_filters; i++) {
        l->filters[i] = make_volume(l->filter_width, l->filter_height,
            l->input_depth, 0.0);
    }

    l->bias = 0.0;
    l->biases = make_volume(1, 1, l->output_depth, l->bias);

    return l;
}


void conv_forward(conv_layer_t *l, volume_t **inputs, volume_t **outputs, int start, int end) {
    // end=1;
#pragma acc kernels present(inputs[start:end-start+1],l,outputs[start:end-start+1])
{
    #pragma acc loop independent 
    for (int i = start; i <= end; i++) {
        volume_t *in = inputs[i];
        volume_t *out = outputs[i];
// #pragma acc data create(in)
        int stride = l->stride;
// printf("Image: %d \n",i);
#pragma acc loop seq
        for(int f = 0; f < l->output_depth; f++) {
            volume_t *filter = l->filters[f];//printf(" filter: %d\n",f);
            int y = -l->pad;
            for(int out_y = 0; out_y < l->output_height; y += stride, out_y++) {
                int x = -l->pad;
                for(int out_x = 0; out_x < l->output_width; x += stride, out_x++) {
                    // printf("Calculating [%d,%d,%d] \n", out_x, out_y, f);
                    // Take sum of element-wise product
                    double sum = 0.0;           
                    for(int fy = 0; fy < filter->height; fy++) {
                        int in_y = y + fy;
                        for(int fx = 0; fx < filter->width; fx++) {
                            int in_x = x + fx;
                            if(in_y >= 0 && in_y < in->height && in_x >=0 && in_x < in->width) {
                                for(int fd = 0; fd < filter->depth; fd++) {
                                    // printf("    (%d,%d,%d)*(%d,%d,%d) sum:%lf\n",fx, fy, fd, in_x, in_y, fd,sum);
                                    sum += volume_get(filter, fx, fy, fd) * volume_get(in, in_x, in_y, fd);
                                    // printf("    %f*%f=%f\n",volume_get(filter, fx, fy, fd),volume_get(in, in_x, in_y, fd),volume_get(filter, fx, fy, fd) * volume_get(in, in_x, in_y, fd));
                                }
                            }
                        }
                    }

                    sum += l->biases->weights[f];//printf("   bias\n");
                    volume_set(out, out_x, out_y, f, sum);
                }
            }
        }
    }
}
    fdump_volume(outputs[0],"outputs_0.txt");fdump_volume(outputs[end],"outputs_1200.txt");
    fdump_volume(outputs[800],"outputs_800.txt");

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

    fclose(fin);
}

network_t *make_network() {
    network_t *net = (network_t *) malloc(sizeof(network_t));

    net->layers[0] = make_volume(32, 32, 3, 0.0);
    net->l0 = make_conv_layer(32, 32, 3, 5, 16, 1, 2);

    net->layers[1] = make_volume(net->l0->output_width, net->l0->output_height, net->l0->output_depth, 0.0);
    return net;
}
//TEST: Change value in volume
volume_t *change_volume(volume_t *new_vol, double value) {
    // volume_t *new_vol = malloc(sizeof(struct volume));
    // new_vol->weights = malloc(sizeof(double) * width * height * depth);

    int width = new_vol->width;
    int height = new_vol->height;
    int depth = new_vol->depth;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int d = 0; d < depth; d++) {
                volume_set(new_vol, x, y, d, value);
            }
        }
    }

    return new_vol;
}

//TEST: Change value in volume on GPU
volume_t *change_volume_acc(volume_t *new_vol, double value) {
    // volume_t *new_vol = malloc(sizeof(struct volume));
    // new_vol->weights = malloc(sizeof(double) * width * height * depth);

    int width = new_vol->width;
    int height = new_vol->height;
    int depth = new_vol->depth;
#pragma acc parallel loop collapse(3) present(new_vol)
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            for (int d = 0; d < depth; d++) {
                volume_set(new_vol, x, y, d, value);
            }
        }
    }

    return new_vol;
}
