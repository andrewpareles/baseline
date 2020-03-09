#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_tile_group_barrier.h>

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
                
                bsg_cuda_print_stat_kernel_start();

                // bsg_print_int(A[127]);

                bsg_cuda_print_stat_start(1);
                int k = 0; // B[k] index
                for (int i = -P; i <= N + P - F; i += S){ //A[i] index of filter in A
                        float val = 0;
                        for (int j = i; j < i + F; j++){ //j is index of filter summation in A
                                if (0 <= j && j < N){ //unpadded region, j - i = 0...F-1
                                        val += filter[F - 1 - (j - i)] * A[j]; //for regular non-flipped filter, use j-i
                                }
                        }

                        B[k] = val;
                        k++;

                }
                bsg_cuda_print_stat_end(1);

                bsg_cuda_print_stat_kernel_end();

                return 0;
        }

}
