#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// #include "layers.h"
#include "volume.h"

const char *DATA_FOLDER = "../../cifar-10-batches-bin";

// // Function to dump the content of a volume for comparison.
// void dump_volume(volume_t* v) {
//     printf("%d,%d,%d", v->width, v->height, v->depth);
//     for (int x = 0; x < v->width; x++) {
//         for (int y = 0; y < v->height; y++) {
//             for (int z = 0; z < v->depth; z++) {
//                 printf(",%.20lf", volume_get(v, x, y, z));
//             }
//         }
//     }
//     printf("\n");
// }


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

    int n = 1200;
    // int start=0, end=n-1;

    batch_t batches[5];
    for (int i = 0; i < 5; i++) {
        batches[i] = NULL;
    }

    printf("Loading batches...\n");
    for (int i = 0; i < n; i++) {
        int batch = i / 10000;
        if (batches[batch] == NULL) {
            batches[batch] = load_batch(batch);
        }
    }

    volume_t **input = (volume_t **) malloc(sizeof(volume_t*)*n);
    for (int i = 0; i < n; i++) {
        input[i] = batches[i / 10000][i % 10000];
    }

    network_t *net = make_network();
    conv_load(net->l0, "../../cs61c/snapshot/layer1_conv.txt");
    volume_t **output = (volume_t **) malloc(sizeof(volume_t*)*n);

    for (int i = 0; i < n; i++) {
        output[i] = make_volume(net->l0->output_width,net->l0->output_height,net->l0->output_depth,0.0);
    }

    //Copy data to GPU
//*
// input
    int in_w=input[0]->width*input[0]->height*input[0]->depth;
    #pragma acc enter data create(input[0:n])
    for(int i=0; i<n;i++){
        #pragma acc enter data create(input[i][0],input[i]->weights[0:in_w])
        #pragma acc update device (input[i]->width,input[i]->height,input[i]->depth)
        #pragma acc update device (input[i]->weights[0:in_w])
    }
// output
int out_w=output[0]->width*output[0]->height*output[0]->depth;
    #pragma acc enter data create(output[0:n])
    for(int i=0; i<n;i++){
        #pragma acc enter data create(output[i][0],output[i]->weights[0:out_w])
        #pragma acc update device (output[i]->width,output[i]->height,output[i]->depth)
        #pragma acc update device (output[i]->weights[0:out_w])
    }
//net
    #pragma acc enter data create(net[0:1],net->l0[0:1],net->layers[0:net->l0->output_depth])
    //net->l0
    #pragma acc update device (net->l0->input_depth, net->l0->input_width,net->l0->input_height)
    #pragma acc update device (net->l0->output_depth, net->l0->output_width,net->l0->output_height,net->l0->bias)
    #pragma acc update device (net->l0->filter_width,net->l0->filter_height,net->l0->stride,net->l0->pad)
    //net->l0->filters
    #pragma acc enter data create(net->l0->filters[0:net->l0->output_depth])
    for(int i=0;i<net->l0->output_depth;i++){
        int we = net->l0->filter_width*net->l0->filter_height*net->l0->input_depth;
        #pragma acc enter data create (net->l0->filters[i][0:1],net->l0->filters[i]->weights[0:we])
        #pragma acc update device (net->l0->filters[i]->width,net->l0->filters[i]->height,net->l0->filters[i]->depth,net->l0->filters[i]->weights[0:we])
    }
    //net->l0->biases
 #pragma acc enter data create (net->l0->biases[0:1],net->l0->biases->weights[0:net->l0->output_depth])
 #pragma acc update device (net->l0->biases->width,net->l0->biases->height,net->l0->biases->depth,net->l0->biases->weights[0:net->l0->output_depth])


 //net->layers[2]
    #pragma acc enter data create (net->layers[0][0:1],net->layers[0]->weights[0:in_w])
 #pragma acc update device (net->layers[0]->width,net->layers[0]->height,net->layers[0]->depth,net->layers[0]->weights[0:in_w])        
    #pragma acc enter data create (net->layers[1][0:1],net->layers[1]->weights[0:out_w])
 #pragma acc update device (net->layers[1]->width,net->layers[1]->height,net->layers[1]->depth,net->layers[1]->weights[0:out_w])        
//*/
// for (int i=0;i<n; i++)
//     dump_volume(output[i]);

    //TEST:
    //  for(int i=0; i<(32*32*16);i++)
    //     output[8]->weights[i] = 8.0;

    // dump_volume(input[8]);
// #pragma acc update host (output[8]->weights[0:32*32*16])
    // dump_volume(output[8]);
//TEST^
// /*
    printf("Conv_Forward\n");

    printf("Images: %d\n",n);
    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t start = 1000000L * tv.tv_sec + tv.tv_usec;

    conv_forward(net->l0,input,output,0,n-1);

    gettimeofday(&tv,NULL);
    uint64_t end = 1000000L * tv.tv_sec + tv.tv_usec;
    // printf("%ld microseconds\n", end - start);
    printf("%ld mseconds\n", (end - start)/1000);

    // dump_volume(input[n-1]);
    // dump_volume(net->l0->filters[15]);

    // for(int i=0; i<n;i++){
    //     dump_volume(output[i]);
    // }

// */
    return 0;
}
