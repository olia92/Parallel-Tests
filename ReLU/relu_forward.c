#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <openacc.h>

#include "volume.h"

#define W 32
#define H 32
#define D 3

int main(){

// Applies the Rectifier Linear Unit (ReLU) function to the input, which sets
// output(x, y, d) to max(0.0, input(x, y, d)).
int n=1200;// Number of Images

//Declare Variables Used
int start=0;
int end=n-1;//1200-1; 

relu_layer_t *l = make_relu_layer(W,H,D);
volume_t **inputs = (volume_t **) malloc(sizeof(volume_t*)*n);
volume_t **outputs = (volume_t **) malloc(sizeof(volume_t*)*n);
#pragma acc enter data create(inputs[0:n],outputs[0:n]) 
//attach(inputs) attach(outputs)


//Create Volumes
for(int i=start;i<=end;i++){
    inputs[i]=make_volume(W,H,D,0.0);
    outputs[i]=make_volume(W,H,D,0.0);
}

//Initialise input
// int counter=0;
for (int i = start; i <= end; i++) {
        for (int x = 0; x < l->input_width; x++) {
            for (int y = 0; y < l->input_height; y++) {
                for (int d = 0; d < l->input_depth; d++) {
                    double value = 2*(rand()/(double) RAND_MAX)-1;//(volume_get(inputs[i], x, y, d) < 0.0) ? 0.0 : volume_get(inputs[i], x, y, d);
                    volume_set(inputs[i], x, y, d, value);
                    }
            }
        }
    }
    for(int i=start; i<=end;i++){
        int we=inputs[i]->width*inputs[i]->height*inputs[i]->depth;
#pragma acc update device(inputs[i]->width,inputs[i]->height,inputs[i]->depth)
#pragma acc update device(outputs[i]->width,outputs[i]->height,outputs[i]->depth)
#pragma acc update device(inputs[i]->weights[0:we])
    }
//     int wet=inputs[0]->width*inputs[0]->height*inputs[0]->depth;
// // #pragma acc update self(inputs[0]->weights[0:wet])
//     fdump_volume(inputs[0],"outputs/inputs_0.txt");
printf("Inistialise\n");

// Relu_Forward
#pragma acc parallel loop default(present)
    for (int i = start; i <= end; i++) {
        #pragma acc loop collapse(3)
        for (int x = 0; x < l->input_width; x++) {
            for (int y = 0; y < l->input_height; y++) {
                for (int d = 0; d < l->input_depth; d++) {
                    // double value = 6.0;//(volume_get(inputs[i], x, y, d) < 0.0) ? 0.0 : volume_get(inputs[i], x, y, d);
                    double value = inputs[i]->weights[((inputs[i]->width * y) + x) * inputs[i]->depth + d];
                    // volume_set(outputs[i], x, y, d, value);
                    // if(value<0.0) value=0.0;
                    value = (value<0.0) ? 0.0 : value;
                    outputs[i]->weights[((outputs[i]->width * y) + x) * outputs[i]->depth + d] = value;
                }
            }
        }
    }
    // for(int i=start;i<=end;i++)
    //     change_volume_acc(outputs[i],9.00);
    printf("Relu forward\n");
for(int i=start; i<=end;i++){
        int we=outputs[i]->width*outputs[i]->height*outputs[i]->depth;
#pragma acc update self(outputs[i]->weights[0:we])
    }
// Check Correctness 
    for (int i = start; i <= end; i++) {
        for (int x = 0; x < l->input_width; x++) {
            for (int y = 0; y < l->input_height; y++) {
                for (int d = 0; d < l->input_depth; d++) {
                    double value = volume_get(outputs[i], x, y, d);
                    assert(0<=value);
                }
            }
        }
    }
    fdump_volume(outputs[3],"outputs/outputs_0.txt");

// Print to Files
    /*
    char file_name[20];
    for(int i=0;i<=end;i++){
        sprintf(file_name, "outputs/input_%d.txt",i);
        fdump_volume(inputs[i],file_name);
    }
    printf("Files...");
    for(int i=0;i<=end;i++){
        sprintf(file_name, "outputs/output_%d.txt",i);
        fdump_volume(outputs[i],file_name);
    }
    printf("Files.\n");
    */
// Compare Memory

    // int flag;
    // int we;
    // for(int i=start; i<end;i++){
    //     assert(inputs[i]->width==outputs[i]->width);
    //     assert(inputs[i]->height==outputs[i]->height);
    //     assert(inputs[i]->depth==outputs[i]->depth);
    
    //     we=inputs[i]->width*inputs[i]->height*inputs[i]->depth;

    //     flag=memcmp(inputs[i]->weights,outputs[i]->weights,we*sizeof(double));
    //     if(flag!=0){ 
    //         printf("!%d\n",i);
    //         return 1;
    //     }
    // }

// The End!
    printf("DONE\n");
        
    return 0;
}
