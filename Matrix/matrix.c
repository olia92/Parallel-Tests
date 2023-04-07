#include <stdio.h>
#include <stdlib.h>


#define W 5
#define H 3
#define D 2

int main(){

int *A=malloc(sizeof(int)*W*H*D);
int B[W][H][D];


for( int i=0; i<W*H*D;i++){
	A[i]=i;

}


for( int i=0; i<W;i++){
	for( int j=0; j<H;j++){
		for( int k=0; k<D;k++){
			int idx=(i*H*D)+(j*D)+k;
			printf("A[%d]=%d \n",idx,A[idx]);
	}}
}
printf("\n");
// for( int i=0; i<W;i++){
// 	for( int j=0; j<H;j++){
// 		for( int k=0; k<D;k++){
// 			printf("B[%d][%d][%d]=%d \n",i,j,k,B[i][j][k]);
// 	}}
// }




return 0;
}


