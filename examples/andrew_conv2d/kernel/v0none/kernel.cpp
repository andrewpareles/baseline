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

#include <cstdlib>
#include <cstring>

// returns a reference to A[x,y]. Instead of A[y*Nx + x], write mat_get(A, x, y, Nx)
float& mat_get(const float *A, const int x, const int y, const int Nx) { 
        return (float&)A[(y*Nx)+x]; 
}

// ignores P and S-- simple version

extern "C" {
        __attribute__((noinline))
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
                int k = 0; // B[k] index
                for (int y = -Py; y <= Ny + Py - Fy; y += Sy){
                        for (int x = -Px; x <= Nx + Px - Fx; x += Sx){
                                float val = 0;
                                for (int fy = y; fy < y + Fy; fy++){ // fy is y index of filter summation in A
                                        if (0 <= y && y <= Ny){
                                                for (int fx = x; fx < x + Fx; fx++){ // fx is x index of filter summation in A
                                                        if (0 <= x && x <= Nx){
                                                                val += mat_get(filter, fx-x, fy-y, Fx) * mat_get(A, fx, fy, Nx);
                                                        }
                                                }
                                }
                                B[k] = val;
                                k++;
                                }
                        }
                }

                return 0;
        }

}
