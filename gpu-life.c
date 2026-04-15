/*
  Name: Evan Poirier
  Email: espoirier@crimson.ua.edu
  Course Section: CS 481
  Homework #: 4
  To Compile: nvcc -O -o gpu-life gpu-life.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DIES   0
#define ALIVE  1

#define BLOCK 16

#define CUDA_CHECK(call) do {                                       \
  cudaError_t _e = (call);                                          \
  if (_e != cudaSuccess) {                                          \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
            __FILE__, __LINE__, cudaGetErrorString(_e));            \
    exit(-1);                                                       \
  }                                                                 \
} while (0)

double gettime(void) {
  struct timeval tval;
  gettimeofday(&tval, NULL);
  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

int **allocarray(int P, int Q) {
  int i, *p, **a;
  p = (int *)malloc(P*Q*sizeof(int));
  a = (int **)malloc(P*sizeof(int*));
  for (i = 0; i < P; i++)
    a[i] = &p[i*Q];
  return a;
}

void freearray(int **a) {
  free(&a[0][0]);
  free(a);
}

void printarray(int **a, int N, int k) {
  int i, j;
  printf("Life after %d iterations:\n", k);
  for (i = 1; i < N+1; i++) {
    for (j = 1; j < N+1; j++)
      printf("%d ", a[i][j]);
    printf("\n");
  }
  printf("\n");
}

// 1 thread per interior cell, stride is N+2 so boundary rows/cols stay zero
__global__ void life_kernel(const int *life, int *temp, int N, int *changed) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > N || j > N) return;

  int stride = N + 2;
  int c = i * stride + j;

  int value = life[c - stride - 1] + life[c - stride] + life[c - stride + 1]
            + life[c - 1]                              + life[c + 1]
            + life[c + stride - 1] + life[c + stride] + life[c + stride + 1];

  int old = life[c];
  int next;
  if (old) next = (value < 2 || value > 3) ? DIES : ALIVE;
  else     next = (value == 3) ? ALIVE : DIES;

  temp[c] = next;
  if (next != old) atomicAdd(changed, 1);
}

// sum of all interior cells
__global__ void count_alive_kernel(const int *life, int N, int *alive) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > N || j > N) return;
  int stride = N + 2;
  if (life[i * stride + j]) atomicAdd(alive, 1);
}

int main(int argc, char **argv) {
  int N, NTIMES, **life = NULL, **temp = NULL;
  int i, j, k, flag = 1, cellsalive = 0;
  double t1, t2;

  if (argc != 3) {
    printf("Usage: %s <size> <max. iterations>\n", argv[0]);
    exit(-1);
  }

  N = atoi(argv[1]);
  NTIMES = atoi(argv[2]);

  life = allocarray(N+2, N+2);
  temp = allocarray(N+2, N+2);

  for (i = 0; i < N+2; i++) {
    life[0][i] = life[i][0] = life[N+1][i] = life[i][N+1] = DIES;
    temp[0][i] = temp[i][0] = temp[N+1][i] = temp[i][N+1] = DIES;
  }

  /* Identical to life.c so both programs produce the same initial board. */
  for (i = 1; i < N+1; i++) {
    srand48(54321+i);
    for (j = 1; j < N+1; j++)
      if (drand48() < 0.5)
        life[i][j] = ALIVE;
      else
        life[i][j] = DIES;
  }

#ifdef DEBUG1
  printarray(life, N, 0);
#endif

  size_t bytes = (size_t)(N+2) * (N+2) * sizeof(int);
  int *d_life = NULL, *d_temp = NULL, *d_changed = NULL;
  CUDA_CHECK(cudaMalloc(&d_life, bytes));
  CUDA_CHECK(cudaMalloc(&d_temp, bytes));
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_life, &life[0][0], bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_temp, 0, bytes));

  dim3 block(BLOCK, BLOCK);
  dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

  t1 = gettime();
  for (k = 0; k < NTIMES && flag != 0; k++) {
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    life_kernel<<<grid, block>>>(d_life, d_temp, N, d_changed);
    CUDA_CHECK(cudaGetLastError());

    int *swap = d_life; d_life = d_temp; d_temp = swap;

    CUDA_CHECK(cudaMemcpy(&flag, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

#ifdef DEBUG2
    CUDA_CHECK(cudaMemcpy(&life[0][0], d_life, bytes, cudaMemcpyDeviceToHost));
    printf("No. of cells whose value changed in iteration %d = %d\n", k+1, flag);
    printarray(life, N, k+1);
#endif
  }

  int *d_alive = NULL;
  CUDA_CHECK(cudaMalloc(&d_alive, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_alive, 0, sizeof(int)));
  count_alive_kernel<<<grid, block>>>(d_life, N, d_alive);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&cellsalive, d_alive, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  t2 = gettime();

#ifdef DEBUG1
  CUDA_CHECK(cudaMemcpy(&life[0][0], d_life, bytes, cudaMemcpyDeviceToHost));
  printarray(life, N, k);
#endif

  printf("Time taken %f seconds for %d iterations, cells alive = %d\n",
         t2 - t1, k, cellsalive);

  cudaFree(d_life);
  cudaFree(d_temp);
  cudaFree(d_changed);
  cudaFree(d_alive);
  freearray(life);
  freearray(temp);

  printf("Program terminates normally\n");
  return 0;
}
