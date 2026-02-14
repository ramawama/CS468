#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

// define these as static, so can free them in teardown
// migth need to change to uint32_t / use a buffer / downsample since atomic add works on 32 bits like ints
// but kernel_bins on dev is uint8_t
static uint8_t *dev_bins = NULL; 
static uint32_t *dev_input = NULL;

__global__ void histo_kernel_gpu(uint32_t *input, uint16_t input_size, uint16_t bin_size);

void opt_2dhisto(uint16_t input_size, uint16_t bin_size)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    // runs 1000 times 
    // launch kernel repeatedly 
    // use cuda memset to reset for each run
    cudaMemset(dev_bins, 0, bins_size*sizeof(uint32_t));
    int threads = 512;
    int blocks = 16; // edit this
    histo_kernel_gpu<<<blocks, threads>>>(dev_input, input_size, bin_size);
}

void opt_2dhisto_setup(uint32_t *input, int h, int w){
    // first allocate the on device 1d matrix (input)
    // and allocate device histogram bins
    // use the static variables declared above
    cudaMalloc((void**)&dev_input, h*w*(sizeof(uint32_t)));
    cudaMalloc((void**)&dev_bins, HISTO_HEIGHT*HISTO_WIDTH*(sizeof(uint8_t)));

    // then memcpy input to device input
    cudaMemcpy(dev_input, input, h*w*(sizeof(uint32_t)), cudaMemcpyHostToDevice);
}   

/* Include below the implementation of any other functions you need */

void opt_2dhisto_teardown(uint8_t *bins){
    // tear down function
    // copy over results to host, free the gpu mem
    cudaMemcpy(bins, dev_bins, HISTO_HEIGHT*HISTO_WIDTH(sizeof(uint8_t)), cudaMemcpyDeviceToHost);
    // now free allocated bins + matrix
    cudaFree(dev_bins);
    cudaFree(dev_input);
}

__global__ void histo_kernel_gpu(uint32_t *input, uint16_t input_size, uint16_t bin_size){
    // implement + account for opts
}