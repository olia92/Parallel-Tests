#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// #include "layers.h"
#include "volume.h"

const char *DATA_FOLDER = "../../cifar-10-batches-bin";

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
    int start=0, end=n-1;

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

    volume_t **output = (volume_t **) malloc(sizeof(volume_t*)*n);
    for (int i = 0; i < n; i++) {
        output[i] = make_volume(32,32,16,0.0);
    }

    network_t *net = make_network();
    conv_load(net->l0, "../../cs61c/snapshot/layer1_conv.txt");

    printf("Conv_Forward\n");

    printf("Images: %d\n",n);
    
    conv_forward(net->l0,input,output,0,n-1);
    // dump_volume(input[n-1]);
    // dump_volume(net->l0->filters[15]);


    return 0;
}