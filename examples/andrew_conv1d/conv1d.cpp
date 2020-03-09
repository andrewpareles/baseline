
#include "conv1d.hpp"

// CODE FOR VALIDATING ANSWER:
template <typename TA, typename TF, typename TB>
void conv1d(const TA *A, const uint32_t A_LENGTH,
            const TF *F, const uint32_t F_LENGTH,
            const uint32_t PAD_LENGTH,
            const uint32_t S_LENGTH,
            TB *B){

        uint32_t M = 1 + (A_LENGTH - F_LENGTH + 2 * PAD_LENGTH) / S_LENGTH;
        for(uint32_t i = 0; i < M; i++)
        {
                uint32_t window_idx = i * S_LENGTH;
                TB res = 0;
                for(uint32_t j = 0; j < F_LENGTH; j++)
                {
                        uint32_t a_idx = window_idx - PAD_LENGTH + j;
                        float a = 0;
                        if(0 <= a_idx && a_idx < A_LENGTH)
                                a = A[a_idx];

                        res += F[F_LENGTH - 1 - j] * a;
                }
                B[i] = res;
        }
}


//HOST CODE:
int kernel_conv1d(int argc, char **argv) {       
        bsg_pr_test_info("Running CUDA Conv1D Kernel.\n\n");
        char *elf, *test_name;
        struct arguments_path args = { NULL, NULL };
        argp_parse(&argp_path, argc, argv, 0, 0, &args);
        elf = args.path;
        test_name = args.name;

        bsg_pr_test_info("name: %s\n",test_name);
        bsg_pr_test_info("elf: %s\n",elf);

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
        
        uint32_t N = 128; //1d image size
        uint32_t F = 7; //1d filter size
        uint32_t P = 8; //padding (symmetric, both sides)
        uint32_t S = 2; //stride
        uint32_t M = 1 + (N - F + 2 * P) / S; //size of output B
        
        size_t A_size = sizeof(float) * N;
        size_t F_size = sizeof(float) * F;
        size_t B_size = sizeof(float) * M;
        
        float A_host[N];
        float filter_host[F];
        float B_result[M];

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

        // create image and filter to convolve:
        for(int i = 0; i < N; i++) {
                A_host[i] = i;
                bsg_pr_test_info("A_host[%d] = %.9f \n", i, A_host[i]);
        }

        for(int i = 0; i < F; i++) {
                filter_host[i] = i;
                bsg_pr_test_info("filter_host[%d] = %.9f \n", i, filter_host[i]);
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
        uint32_t cuda_argv[] = { A_device, N, filter_device, F, P, B_device, S };
        size_t cuda_argc = 7; // # args = 7
        // data/hb/bsg_bladerunner/bsg_replicant/libraries/bsg_manycore_cuda.h

        //load kernel code onto device
        rc = hb_mc_kernel_enqueue(device, grid_dim, tilegroup_dim, "kernel_conv1d", cuda_argc, cuda_argv);
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
        float B_expected[M];
        conv1d(A_host, N, filter_host, F, P, S, B_expected);

        // compare result to expected 
        for(int i = 0; i < M; i++) {
                bsg_pr_test_info("B_result[%d] = %.4f,\t\texpected B_result[i] =%.4f\n", i, B_result[i], B_expected[i]);
                if (B_result[i] != B_expected[i]) return HB_MC_FAIL;
        }

        return HB_MC_SUCCESS;
}
//ERROR WRAPPER
// int wrap(int &rc, int (*f)(const uint32_t *argv), const uint32_t *argv, uint32_t argc, char *err_msg){
//         rc = f(argv);
// }



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
        int rc = kernel_conv1d(argc, argv);
        *exit_code = rc;
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return;
}
#else
int main(int argc, char **argv)
{
        int rc = kernel_conv1d(argc, argv);
        bsg_pr_test_pass_fail(rc == HB_MC_SUCCESS);
        return rc;
}
#endif