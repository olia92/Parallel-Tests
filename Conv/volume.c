#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

// Include OpenMP
#include <omp.h>
#include <openacc.h>

#include "volume.h"

int vol_count=0;
#pragma acc routine seq
inline double volume_get(volume_t *v, int x, int y, int d) {
    return v->weights[((v->width * y) + x) * v->depth + d];
}

#pragma acc routine seq
inline void volume_set(volume_t *v, int x, int y, int d, double value) {
    v->weights[((v->width * y) + x) * v->depth + d] = value;
}

volume_t *make_volume(int width, int height, int depth, double value) {
    volume_t *new_vol = malloc(sizeof(struct volume));vol_count++;
#pragma acc enter data create(new_vol[0:1])
    new_vol->weights = malloc(sizeof(double) * width * height * depth);
#pragma acc enter data create(new_vol->weights[0:(width * height * depth)])
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
#pragma acc update device(new_vol->width,new_vol->height,new_vol->depth,new_vol->weights[0:(width * height * depth)])
    return new_vol;
}


void copy_volume(volume_t *dest, volume_t *src) {
    assert(dest->width == src->width);
    assert(dest->height == src->height);
    assert(dest->depth == src->depth);
    
#pragma acc parallel loop collapse(3) default(present)
    for (int x = 0; x < dest->width; x++) {
        for (int y = 0; y < dest->height; y++) {
            for (int d = 0; d < dest->depth; d++) {
                volume_set(dest, x, y, d, volume_get(src, x, y, d));
            }
        }
    }

//This When copy volume on the host ->
// #pragma acc update device(dest->weights[0:(dest->width * dest->height * dest->depth)])

//TEST: This Copy Volme to Device
#pragma acc update host(dest->weights[0:(dest->width * dest->height * dest->depth)])

// //TEST:->
// change_volume(dest,8.00);

// #pragma acc update self(dest->width,dest->height,dest->depth,dest->weights[0:(dest->width * dest->height * dest->depth)])
// #pragma acc update self(dest->weights[0:(dest->width * dest->height * dest->depth)])

// fdump_volume(dest,"output/dest.txt");
//TEST:^
}

//TEST:Copy Volume Host
void copy_volume_host(volume_t *dest, volume_t *src) {
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
#pragma acc exit data delete(v->weights[0:(v->height*v->width*v->depth)])
#pragma acc exit data delete(v)
    free(v);
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
// #pragma acc routine
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
