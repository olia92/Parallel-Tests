#include <stdlib.h>
#include <stdio.h>

void saxpy (int n, float a, float *x, float *restrict y){
    #pragma acc parallel loop
    for (int i = 0; i< n; ++i)
        y[i] = a * x[i] + y[i];
}

int main( int argc, char **argv){
    int N = 1<<20; // 1 million floats

    if (argc>1)
        N = atoi(argv[1]);

    printf("N=%d\n",N);
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));

    for (size_t i = 0; i < N; i++){
        x[i] = 2.0f;
        y[i] = 1.0f;
    }
    
    saxpy(N, 3.0,x,y);

    // for(int i=0; i<N;i++)
    //     printf("%f ",y[i]);
    return 0;
}