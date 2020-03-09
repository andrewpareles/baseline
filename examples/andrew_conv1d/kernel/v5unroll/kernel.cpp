#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_tile_group_barrier.h>

#include <cstdlib>
#include <cstring>

int max(int x, int y){ return x >= y ? x : y; }
int min(int x, int y){ return x <= y ? x : y; }

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

                //copy A to array
                float array[N];
                memcpy(array, A, N*sizeof(float));
                A = array;

                //flip filter
                float filterFlip[F];
                for (int i = 0; i < F; i++){
                        filterFlip[i] = filter[F - i - 1];
                }
                filter = filterFlip;
                
                bsg_cuda_print_stat_start(1);
                int k = 0; // B[k] index
                for (int i = -P; i <= N + P - F; i += S){ //A[i] index of filter in A
                        float val = 0;
                        for (int j = max(i, 0); j < min(i + F, N); j++) { 
                                // if j is outside the range 0 <= j < L, then we're in padded zone,
                                // so the value gets zeroed out. only care about values in range
                                val += filter[F - 1 - (j - i)] * A[j];
                        }

                        B[k] = val;
                        k++;
                }
                bsg_cuda_print_stat_end(1);

                bsg_cuda_print_stat_kernel_end();

                return 0;
        }

}
