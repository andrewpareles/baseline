
#include "conv2d.hpp"

#include <iostream>

#define PRINT_DEBUG 0

// returns a reference to A[x,y]. Instead of A[y*Nx + x], write mat_get(A, x, y, Nx)
float& mat_get(const float *A, const int x, const int y, const int Nx) { 
        return (float&)A[(y*Nx)+x]; 
}



// Print matrix A (M x N). This works well for small matricies.
template <typename T>
void matrix_print(T *A, uint64_t M, uint64_t N) {
        T sum = 0;
        for (uint64_t y = 0; y < M; y ++) {
                for (uint64_t x = 0; x < N; x ++) {
                        std::cout << A[y * N + x] << "\t";
                }
                std::cout << '\n';

        }
}

void conv2d(const float *A, //array to convolve
        const int Nx, //A xlen
        const int Ny, //A ylen
        const float *filter,
        const int Fx, //filter xlen
        const int Fy, //filter ylen
        const int Px, //pad xlen (on both sides of A)
        const int Py, //pad ylen (on both sides of A)
        const int Sx, //ystride
        const int Sy, //xstride
        float *B // answer array
        ) { 

        //copy over filter
        int F = Fx*Fy;
        float filterarray[F];
        memcpy(filterarray, filter, F*sizeof(float));

        //flip filter
        for (int i = 0; i < F/2; i++){ // flip using a neat trick
                float temp = filterarray[i];
                filterarray[i] = filterarray[F-i-1];
                filterarray[F-i-1] = temp;
        }
        filter = filterarray; // update pointer
        
        printf("Flipped filter:\n");
        matrix_print(filter, Fy, Fx);

        int k = 0; // B[k] index
        for (int y = -Py; y <= Ny + Py - Fy; y += Sy){
                for (int x = -Px; x <= Nx + Px - Fx; x += Sx){
                        float val = 0;
                        for (int fy = y; fy < y + Fy; fy++){ // fy is y index of filter summation in A
                                if (0 <= y && y <= Ny){
                                        for (int fx = x; fx < x + Fx; fx++){ // fx is x index of filter summation in A
                                                if (0 <= x && x <= Nx){
                                                        // printf("filter[%i,%i]=%f\tA[%i, %i]=%f\n", fy-y, fx-x, mat_get(filter, fx-x, fy-y, Fx), fy, fx, mat_get(A, fx, fy, Nx));
                                                        val += mat_get(filter, fx-x, fy-y, Fx) * mat_get(A, fx, fy, Nx);
                                                }
                                        }
                                }
                        }
                        // printf("\n");
                        B[k] = val;
                        k++;
                }
        }
}

//HOST CODE:
int kernel_conv2d(int argc, char **argv) {       
        bsg_pr_test_info("Running CUDA Conv2D Kernel.\n\n");
        char *elf, *test_name;
        struct arguments_path args = { NULL, NULL };
        argp_parse(&argp_path, argc, argv, 0, 0, &args);
        elf = args.path;
        test_name = args.name;

        if (PRINT_DEBUG){
                bsg_pr_test_info("name: %s\n",test_name);
                bsg_pr_test_info("elf: %s\n",elf);
        }

        int rc;
        hb_mc_device_t temp; 
        hb_mc_device_t *device = &temp;

        // init device
        rc = hb_mc_device_init(device, test_name, 0);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to initialize device.\n");
                return rc;
        }

        // init program
        rc = hb_mc_device_program_init(device, elf, "default_allocator", 0);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to initialize the program.\n");
                return rc;
        }
        
        uint32_t Nx = 3; //2d image x-size
        uint32_t Ny = 7; //2d image y-size

        uint32_t Fx = 2; //2d filter x-size
        uint32_t Fy = 3; //2d filter x-size
        
        uint32_t Px = 0; //x-padding (symmetric, both sides)
        uint32_t Py = 0; //y-padding (symmetric, both sides)

        uint32_t Sx = 1; //x-stride
        uint32_t Sy = 1; //y-stride

        uint32_t Mx = 1 + (Nx - Fx + 2 * Px) / Sx; //x-size of output B
        uint32_t My = 1 + (Ny - Fy + 2 * Py) / Sy; //y-size of output B
        
        size_t A_size = sizeof(float) * Nx * Ny;
        size_t F_size = sizeof(float) * Fx * Fy;
        size_t B_size = sizeof(float) * Mx * My;
        
        float A_host[Nx * Ny];
        float filter_host[Fx * Fy];
        float B_result[Mx * My];

        // create image and filter to convolve:
        for (int y = 0; y < Ny; y++){
                for(int x = 0; x < Nx; x++) {
                        mat_get(A_host, x, y, Nx) = (x + 1) + 2 * (y + 1);
                }
        }
        bsg_pr_test_info("A_host = \n");
        matrix_print(A_host, Ny, Nx);

        //{1, -2, 0, 1, 1, 0}
        filter_host[0] = 1;
        filter_host[1] = -2;
        filter_host[3] = 1;
        filter_host[4] = 1;
        bsg_pr_test_info("filter_host = \n");
        matrix_print(filter_host, Fy, Fx);
        

        //memory allocation on device
        eva_t A_device, B_device, filter_device;
        rc = hb_mc_device_malloc(device, A_size, &A_device);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to allocate A on the manycore.\n");
                return rc;
        }

        rc = hb_mc_device_malloc(device, F_size, &filter_device);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to allocate F on the manycore.\n");
                return rc;
        }
        
        rc = hb_mc_device_malloc(device, B_size, &B_device);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to allocate B on the manycore.\n");
                return rc;
        }

        //put A and filter on device:
        rc = hb_mc_device_memcpy(device,
                                 (void *) ((intptr_t) A_device),
                                 (void *) &A_host[0],
                                 A_size, HB_MC_MEMCPY_TO_DEVICE);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to copy A to the manycore.\n");
                return rc;
        }
        
        rc = hb_mc_device_memcpy(device, (void *) ((intptr_t) filter_device), 
                                 (void *) &filter_host[0], 
                                 F_size, HB_MC_MEMCPY_TO_DEVICE);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to copy F to the manycore.\n");
                return rc;
        }

        hb_mc_dimension_t tilegroup_dim = { .x = 1, .y = 1 };
        hb_mc_dimension_t grid_dim      = { .x = 1, .y = 1 };
        uint32_t cuda_argv[] = { A_device, Nx, Ny, filter_device, Fx, Fy, Px, Py, B_device, Sx, Sy };
        size_t cuda_argc = 11; // # args = 11

        //load kernel code onto device
        rc = hb_mc_kernel_enqueue(device, grid_dim, tilegroup_dim, "kernel_conv2d", cuda_argc, cuda_argv);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to initialize grid.\n");
                return rc;
        }

        //run kernel code on device
        rc = hb_mc_device_tile_groups_execute(device);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to execute tilegroups.\n");
                return rc;
        }

        //copy result from device to host
        rc = hb_mc_device_memcpy(device, (void *) B_result, 
                                 (void *) ((intptr_t) B_device), 
                                 B_size, HB_MC_MEMCPY_TO_HOST);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to copy result to host.\n");
                return rc;
        }

        // "finish"?
        rc = hb_mc_device_finish(device);
        if(rc != HB_MC_SUCCESS) {
                bsg_pr_test_err("Failed to deinitialize the manycore.\n");
                return rc;
        }

        // compute expected value
        float B_expected[Mx*My];
        conv2d(A_host, Nx, Ny, filter_host, Fx, Fy, Px, Py, Sx, Sy, B_expected);

        float B_diff[Mx*My];
        // compare result to expected 
        float sse = 0;
        for(int y = 0; y < My; y++) {
                for(int x = 0; x < Mx; x++) {
                        float diff = mat_get(B_result, x, y, Mx) - mat_get(B_expected, x, y, Mx);
                        mat_get(B_diff, x, y, Mx) = diff;
                        sse += diff * diff;                
                        bsg_pr_test_info("B_result[%d, %d] = %f,\tB_expected[%d, %d] = %f\tdiff = %f\n", y, x, mat_get(B_result, x, y, Mx), y, x, mat_get(B_expected, x, y, Mx), diff);
                }
        }
        bsg_pr_test_info("B_result = \n");
        matrix_print(B_result, My, Mx);
        
        bsg_pr_test_info("B_expected = \n");
        matrix_print(B_expected, My, Mx);

        bsg_pr_test_info("B_diff = \n");
        matrix_print(B_diff, My, Mx);

        bsg_pr_test_info("SSE between B_result and B_expected: %f\n", sse);

        return HB_MC_SUCCESS;
}


// MAIN (RUN HOST CODE)
#ifdef COSIM
void cosim_main(uint32_t *exit_code, char *args)
{
        // We aren't passed command line arguments directly so we parse them
        // from *args. args is a string from VCS - to pass a string of arguments
        // to args, pass c_args to VCS as follows: +c_args="<space separated
        // list of args>"
        int argc = get_argc(args);
        char *argv[argc];
        get_argv(args, argc, argv);

#ifdef VCS
        svScope scope;
        scope = svGetScopeFromName("tb");
        svSetScope(scope);
#endif
        int rc = kernel_conv2d(argc, argv);
        *exit_code = rc;
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return;
}
#else
int main(int argc, char **argv)
{
        int rc = kernel_conv2d(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}
#endif