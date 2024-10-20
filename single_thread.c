#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 100
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
  for (int i = 0; i < SIZE; i++) {
   int value = 0;
   for (int j = 0; j <= i; j++) {
     value += input[j];
   }
    output[i] = value;
  }

  double t0,t1;
  t0 = get_clock();
  // check results
  for (int i = 0; i < SIZE; i++) {
    printf("%d ", output[i]);
  }
  printf("\n");
  t1=get_clock();
  printf("time per call: %f ns\n", (1000000000.0*(t1-t0)/SIZE) );

  // free mem
  free(input);
  free(output);

  return 0;
}
