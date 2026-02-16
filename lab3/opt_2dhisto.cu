#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cmath>
#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

// define these as static, so can free them in teardown
// migth need to change to uint32_t / use a buffer / downsample since atomic add works on 32 bits like ints
// but kernel_bins on dev is uint8_t
static uint8_t *dev_bins_8 = NULL; 
static uint32_t *dev_bins_32 = NULL; 
static uint32_t *dev_input = NULL;

__global__ void histo_kernel_gpu(const uint32_t * __restrict__ input, uint32_t * __restrict__ bins_32, uint32_t input_size, uint16_t bin_size);

__global__ void bin_reduction_gpu(const uint32_t * __restrict__ bins_32, uint8_t * __restrict__ bins_8, uint16_t bin_size);

void opt_2dhisto(uint32_t input_size, uint16_t bin_size)
{
    int threads = 512;

    //since four pixles per thread 
    int blocks = ceil((double)(input_size / 4.0) / threads);

    int blocks_reduction = ceil((double)bin_size / threads);

    cudaMemset(dev_bins_32, 0, bin_size * sizeof(uint32_t));

    histo_kernel_gpu<<<blocks, threads>>>(dev_input, dev_bins_32, input_size, bin_size);

    // Ensure histogram is finished before reduction
    // cudaDeviceSynchronize();

    bin_reduction_gpu<<<blocks_reduction, threads>>>(dev_bins_32, dev_bins_8, bin_size);
}

void opt_2dhisto_setup(uint32_t *input, int h, int w){

    cudaMalloc((void**)&dev_input, h*w*(sizeof(uint32_t)));
    cudaMalloc((void**)&dev_bins_8, HISTO_HEIGHT*HISTO_WIDTH*(sizeof(uint8_t)));
    cudaMalloc((void**)&dev_bins_32, HISTO_HEIGHT*HISTO_WIDTH*(sizeof(uint32_t)));

    cudaMemset( dev_bins_8, (int)0, HISTO_HEIGHT*HISTO_WIDTH*(sizeof(uint8_t))); //the bins should be set to 0
    cudaMemset( dev_bins_32, (int)0, HISTO_HEIGHT*HISTO_WIDTH*(sizeof(uint32_t))); //the bins should be set to 0

    cudaMemcpy(dev_input, input, h*w*(sizeof(uint32_t)), cudaMemcpyHostToDevice);
}   

/* Include below the implementation of any other functions you need */

void opt_2dhisto_teardown(uint8_t *bins){
    cudaMemcpy( bins, dev_bins_8, HISTO_HEIGHT*HISTO_WIDTH*(sizeof(uint8_t)),  cudaMemcpyDeviceToHost);

    cudaFree(dev_bins_8);
    cudaFree(dev_bins_32);
    cudaFree(dev_input);
    
 
}


__global__ void histo_kernel_gpu(const uint32_t * __restrict__ input, 
                                 uint32_t * __restrict__ bins_32, 
                                 uint32_t input_size, 
                                 uint16_t bin_size) {
    __shared__ uint32_t local_hist[HISTO_HEIGHT * HISTO_WIDTH];
    
    #pragma unroll 4  
    for (int i = threadIdx.x; i < bin_size; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    uint32_t global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t vec_limit = input_size / 4; 

    if (global_idx < vec_limit) {
        uint4 data = ((uint4*)input)[global_idx];  //get 4 pixles

        uint32_t x0 = data.x & ((1 << HISTO_LOG) - 1); 
        uint32_t y0 = data.x >> HISTO_LOG;             
        uint32_t bin0 = y0 * HISTO_WIDTH + x0;         
        atomicAdd(&local_hist[bin0], 1);

        uint32_t x1 = data.y & ((1 << HISTO_LOG) - 1);
        uint32_t y1 = data.y >> HISTO_LOG;
        uint32_t bin1 = y1 * HISTO_WIDTH + x1;
        atomicAdd(&local_hist[bin1], 1);

        uint32_t x2 = data.z & ((1 << HISTO_LOG) - 1);
        uint32_t y2 = data.z >> HISTO_LOG;
        uint32_t bin2 = y2 * HISTO_WIDTH + x2;
        atomicAdd(&local_hist[bin2], 1);

        uint32_t x3 = data.w & ((1 << HISTO_LOG) - 1);
        uint32_t y3 = data.w >> HISTO_LOG;
        uint32_t bin3 = y3 * HISTO_WIDTH + x3;
        atomicAdd(&local_hist[bin3], 1);
    }


    if (global_idx == 0 && (input_size % 4 != 0)) {
        for (uint32_t i = vec_limit * 4; i < input_size; i++) {
             uint32_t pixel = input[i];
             atomicAdd(&local_hist[(pixel >> HISTO_LOG) * HISTO_WIDTH + (pixel & ((1 << HISTO_LOG) - 1))], 1);
        }
    }

    __syncthreads();

    #pragma unroll 4
    for (int i = threadIdx.x; i < bin_size; i += blockDim.x) {
        uint32_t count = local_hist[i];
        if (__any_sync(0xFFFFFFFF, count != 0)) { 
            // (mask, predicate)
            // if any thread in the warp has non-zero value, enter block
            // updates if there are non-zero values
            atomicAdd(&bins_32[i], count);
        }
    }
}





__global__ void bin_reduction_gpu(const uint32_t * __restrict__ bins_32, 
                                  uint8_t * __restrict__ bins_8, 
                                  uint16_t bin_size) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= bin_size) return;

    // #pragma unroll 4
    // for (uint32_t i = idx; i < bin_size; i += blockDim.x) {
        bins_8[idx] = (bins_32[idx] >= 255) ? 255 : (uint8_t)bins_32[idx];
    // }

}

