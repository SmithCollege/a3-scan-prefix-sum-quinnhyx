#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 1000000
double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
  // allocate memory
  int* input = malloc(sizeof(int) * SIZE);
  int* output = malloc(sizeof(int) * SIZE);

  // initialize inputs
  for (int i = 0; i < SIZE; i++) {
    input[i] = 1;
   }

  // do the scan
  double t0,t1;
  t0=get_clock();
  output[0]=input[0];
  for (int i = 1; i < SIZE; i++) {
    output[i] = output[i-1]+input[i];
  }
  t1=get_clock();
   // check results
  for (int i = 0; i < SIZE; i++) {
    printf("%d ", output[i]);
  }
  printf("\n");
  
  printf("time per call: %f ns\n", (1000000000.0*(t1-t0)) );

  // free mem
  free(input);
  free(output);

  return 0;
}
