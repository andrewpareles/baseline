// BSG_TILE_GROUP_X_DIM and BSG_TILE_GROUP_Y_DIM must be defined
// before bsg_manycore.h and bsg_tile_group_barrier.h are
// included. bsg_tiles_X and bsg_tiles_Y must also be defined for
// legacy reasons, but they are deprecated.
#define BSG_TILE_GROUP_X_DIM 1
#define BSG_TILE_GROUP_Y_DIM 1
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>

#include <cstdlib>
#include <cstring>

#define a(A, x, y, Ny) A[(y*Ny) + x]
// ignores P and S-- simple version

extern "C" {
        __attribute__((noinline))
        // A_device, Nx, Ny, filter_device, Fx, Fy, Px, Py, B_device, Sx, Sy
        int kernel_conv2d(const float *A, //array to convolve
                          const int Nx, //A xlen
                          const int Ny, //A ylen
                          const float *filter,
                          const int Fx, //filter xlen
                          const int Fy, //filter ylen
                          const int Px, //pad xlen (on both sides of A)
                          const int Py, //pad ylen (on both sides of A)
                          float *B, // answer array
                          const int Sx, //ystride
                          const int Sy) { //xstride
                // ignores P and S-- simple version
                bsg_cuda_print_stat_kernel_start();
                bsg_cuda_print_stat_start(1);
                
                int k = 0; // B[k] index
                for (int y = 0; y <= Ny - Fy; y++){
                        for (int x = 0; x <= Nx - Fx; x++){
                                float val = 0;
                                for (int fy = y; fy < y + Fy; y++){ // fy is y index of filter summation in A
                                        for (int fx = x; fx < x + Fx; x++){ // fx is x index of filter summation in A
                                                val += a(filter, Fx-1-(fx-x), Fy-1-(fy-y), Fy) * a(A, fx, fy, Ny);
                                        }
                                }
                                B[k] = val;
                                k++;
                        }
                }
                bsg_cuda_print_stat_end(1);
                bsg_cuda_print_stat_kernel_end();

                return 0;
        }

}
