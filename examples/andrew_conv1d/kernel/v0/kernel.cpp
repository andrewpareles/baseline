// Takes a vector A of length N and a 1D filter of size F, padding
// size P, and stride S.  Performs 1D convolution of A with the filter
// and stores the result in B of size M = 1 + (N - F + 2P) / S.

// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROU`P_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>

#include <cstdlib>
#include <cstring>

extern "C" {
        __attribute__((noinline))
        int kernel_conv1d(const float *A, //array to convolve
                          const int N, //A len
                          const float *filter, 
                          const int F, //filter len
                          const int P, //pad len (on both sides of A)
                          float *B, // answer array
                          const int S) { //stride, # elts you skip each step

                int k = 0; // B[k] index
                for (int i = -P; i <= N + P - F; i += S){ //A[i] index of filter in A
                        float val = 0;

                        //copy A to array
                        float array[N];
                        memcpy(array, A, N*sizeof(float));

                        // loop unrolling with removal of if statement?
                        // act as host and call another kernel function to speed this up?
                        //worth it to flip filter at beginning once to reduce # instructions?
// #pragma GCC unroll 8
                        for (int j = i; j < i + F; j++){ //j is index of filter summation in A
                                if (0 <= j && j < N){ //unpadded region
                                        val += filter[F - 1 - (j - i)] * array[j];
                                } else { //paddded region
                                //...
                                }
                        }

                        // ALTERNATIVELY:
//                         #define max(x,y) x >= y ? x : y
//                         #define min(x,y) x <= y ? x : y
// #pragma GCC unroll 8
//                         for (int j = max(i, 0); j < min(i + F, N); j++){ //j is index of filter summation in A
//                              val += filter[F - 1 - (j - i)] * A[j];
//                         }

//                         B[k] = val;
                        k++;
                }
                return 0;
        }

}
