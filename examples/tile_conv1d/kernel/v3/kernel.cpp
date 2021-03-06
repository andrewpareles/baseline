// Takes a vector A of length N and a 1D filter of size F, padding
// size P, and stride S.  Performs 1D convolution of A with the filter
// and stores the result in B of size M = 1 + (N - F + 2P) / S.

// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_tile_group_barrier.h>
#include <cstdint>
#include <cstring>


template <uint32_t FACTOR, typename TI, typename TF, typename TO>
int conv1d(const TI *INPUT,
           const uint32_t i_nelements,
           const TF *FILTER,
           const uint8_t f_nelements,
           TO *OUTPUT){

                uint32_t nstep = f_nelements - FACTOR;
                uint32_t b_nelements = i_nelements - f_nelements + 1;
                uint32_t ii = 0;
                for(uint32_t oi = 0; oi < b_nelements; oi += FACTOR){
                        TO sum[FACTOR];
                        for (uint32_t fi = 0; fi < FACTOR; ++fi){
                                asm volatile ("fmv.s.x %0,zero\n\t" : "=f" (sum[fi]));
                        }
                        TI a[FACTOR];

                        for(uint32_t fi = 0; fi < f_nelements; fi++) {
                                TF f = FILTER[fi];

#pragma GCC unroll 8
                                for(uint32_t ui = 0; ui < FACTOR; ++ui){
                                        a[ui] = INPUT[ii + ui + fi];
                                }

#pragma GCC unroll 8
                                for(uint32_t ui = 0; ui < FACTOR; ++ui){
                                        sum[ui] += f * a[ui];
                                }

                        }
                        
#pragma GCC unroll 8
                        for(uint32_t ui = 0; ui < FACTOR; ++ui){
                                OUTPUT[oi + ui] = sum[ui];
                        }

                        ii += FACTOR;
                }

                return 0;
}


extern "C" {
        __attribute__((noinline))
        int kernel_tile_conv1d(const float *INPUT,
                               const int i_nelements,
                               const float *FILTER,
                               const int f_nelements,
                               float *OUTPUT)
        {
                int rc;
                uint32_t o_nelements = i_nelements - f_nelements + 1;

                float input[i_nelements];
                float filter[f_nelements];
                float output[o_nelements];

                memcpy (input, INPUT, sizeof(INPUT[0])*i_nelements);
                memcpy (filter, FILTER, sizeof(FILTER[0])*f_nelements);

                for(int i = 0; i < 2; ++i){
                        bsg_cuda_print_stat_start(i);
                        rc = conv1d<4>(input, i_nelements, filter, f_nelements, output);
                        bsg_cuda_print_stat_end(i);
                }

                memcpy (OUTPUT, output, sizeof(OUTPUT[0])*o_nelements);

                return rc;
        }
}
