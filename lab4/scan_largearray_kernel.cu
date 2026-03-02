#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define BLOCK_SIZE 256

#define ELEMENTS_PER_BLOCK (BLOCK_SIZE * 2)

// adds padding offset to avoid shared memory bank conflicts.
#define CONFLICT_FREE_OFFSET(n) ((n) / NUM_BANKS)

// padded shared memory size to avoid bank conflicts
#define SHARED_MEM_SIZE (ELEMENTS_PER_BLOCK + (ELEMENTS_PER_BLOCK / NUM_BANKS) + 1)

// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void scanBlock(float *outArray, float *inArray, float *blockSums, int numElements) {
    // use shared mem to scan each block reduce bank conflicts
    // use reduction to compute partial sums of the array
    // then an exclusive scan to compute the prefix sums for each block
    // swap+add method

    // extra padding slots eliminate bank conflicts during strided access
    __shared__ float sharedMem[SHARED_MEM_SIZE];
    int threadId = threadIdx.x;
    int leftIndex  = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    // populate shared memory with padded indices for this thread's two elements
    int leftSharedIndex  = threadId + CONFLICT_FREE_OFFSET(threadId);
    int rightSharedIndex = threadId + BLOCK_SIZE + CONFLICT_FREE_OFFSET(threadId + BLOCK_SIZE);

    // each thread loads 2 elements
    // pad 0 for out of bounds
    if (leftIndex < numElements) {
        sharedMem[leftSharedIndex] = inArray[leftIndex];
    } else {
        sharedMem[leftSharedIndex] = 0.0f;
    }
    if (rightIndex < numElements) {
        sharedMem[rightSharedIndex] = inArray[rightIndex];
    } else {
        sharedMem[rightSharedIndex] = 0.0f;
    }
    // sync threads to ensure shared mem is populated
    __syncthreads();

    // compute partial sums now with reduction
    int stride = 1;
    while (stride < ELEMENTS_PER_BLOCK) {
        // get index to update for this stride
        // each thread updates one elem in sm
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            // if in bounds, add val using padded indices
            int left  = index - stride;
            sharedMem[index + CONFLICT_FREE_OFFSET(index)] += sharedMem[left + CONFLICT_FREE_OFFSET(left)];
        }
        stride *= 2;
        __syncthreads();
    }

    // exclusive scan w swap+add
    // set last elem to 0, since it will be propagated
    if (threadId == 0) {
        // store first, then set to 0
        int last = ELEMENTS_PER_BLOCK - 1;
        int lastPadded = last + CONFLICT_FREE_OFFSET(last);
        if (blockSums != NULL)
            blockSums[blockIdx.x] = sharedMem[lastPadded];
        sharedMem[lastPadded] = 0.0f;
    }
    __syncthreads();

    // stride = blocksize since we are going block by block
    stride = BLOCK_SIZE;
    while (stride > 0) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            // left child = index-stride, right child = index
            // swap, then add left child to right child using padded indices
            int left = index - stride;
            int leftPadded = left  + CONFLICT_FREE_OFFSET(left);
            int rightPadded = index + CONFLICT_FREE_OFFSET(index);
            float left_val = sharedMem[leftPadded];
            sharedMem[leftPadded] = sharedMem[rightPadded];
            sharedMem[rightPadded] += left_val;
        }
        // divide by 2 to swap+add next lvl / inside of block
        stride /= 2;
        __syncthreads();
    }

    // write results to output from sm using same padded indices as load
    if (leftIndex  < numElements) {
        outArray[leftIndex]  = sharedMem[leftSharedIndex];
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] = sharedMem[rightSharedIndex];
    }
}

__global__ void addSums(float *outArray, float *blockSums, int numElements) {
    // add the block sums to every block i+1 -> n
    int threadId = threadIdx.x;
    int leftIndex = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    float blockSum = blockSums[blockIdx.x];
    if (leftIndex < numElements) {
        outArray[leftIndex] += blockSum;
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] += blockSum;
    }
} 

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{

    // 1. Initial scan of each block
    // 2. Scan the block sums and store in subArr
    // 3. Scan block sums and add to each scanned block i+1 -> n

    // three kernel calls, one for each step

    // for initial scan
    int numBlocks = (numElements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    // for scanning block sums
    int numSubBlocks = (numBlocks + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    
    // allocate arrays for block sums + prefix scan of block sums
    float *dev_block_sums;
    cudaMalloc((void**)&dev_block_sums, numBlocks * sizeof(float));
    float *dev_block_prefix_sums;
    cudaMalloc((void**)&dev_block_prefix_sums, numBlocks * sizeof(float));

    // allocate subArray for group of block sums + prefix scan of group sums
    // subArray = group of blocks
    float *dev_sub_sums;
    cudaMalloc((void**)&dev_sub_sums, numSubBlocks * sizeof(float));
    float *dev_sub_prefix_sums;
    cudaMalloc((void**)&dev_sub_prefix_sums, numSubBlocks * sizeof(float));

    // 1. Scan all blocks, store indiv. block sum in dev_block_sums
    scanBlock<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, dev_block_sums, numElements);

    // 2. Scan dev_block_sums + store the prefix sums in dev_block_prefix_sums
    scanBlock<<<numSubBlocks, BLOCK_SIZE>>>(dev_block_prefix_sums, dev_block_sums, dev_sub_sums, numBlocks);

    // 3. Scan dev_sub_sums and store the prefix sums in dev_sub_prefix_sums
    scanBlock<<<1, BLOCK_SIZE>>>(dev_sub_prefix_sums, dev_sub_sums, NULL, numSubBlocks);

    // add prefix sums of sub sums to each block prefix sum
    // then add block prefix sums to each index
    addSums<<<numSubBlocks, BLOCK_SIZE>>>(dev_block_prefix_sums, dev_sub_prefix_sums, numBlocks);
    addSums<<<numBlocks, BLOCK_SIZE>>>(outArray, dev_block_prefix_sums, numElements);

    cudaFree(dev_block_sums);
    cudaFree(dev_block_prefix_sums);
    cudaFree(dev_sub_sums);
    cudaFree(dev_sub_prefix_sums);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
