#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <assert.h>
#include <cuda_runtime.h>

#if defined(PRESCAN_V9_THRUST)
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define BLOCK_SIZE 256
#define ELEMENTS_PER_BLOCK (BLOCK_SIZE * 2)
#define ELEMENTS_PER_BLOCK4 (BLOCK_SIZE * 4)

#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

#if (defined(PRESCAN_V0_NAIVE) + \
     defined(PRESCAN_V1_WITH_SHARED_MEMORY) + \
     defined(PRESCAN_V2_PADDED_BANK_CONFLICT) + \
     defined(PRESCAN_V3_UNROLLED) + \
     defined(PRESCAN_V4_BRANCHLESS) + \
     defined(PRESCAN_V5_HILLIS_STEELE) + \
     defined(PRESCAN_V6_RECURSIVE) + \
     defined(PRESCAN_V7_BUFFER_REUSE) + \
     defined(PRESCAN_V8_FOUR_ELEMS_PER_THREAD) + \
     defined(PRESCAN_V9_THRUST) + \
     defined(PRESCAN_V10_LAUNCH_BOUNDS) + \
     defined(PRESCAN_V11_CACHE_CONFIG)) > 1
#error "Define only one PRESCAN_V* macro at a time"
#endif

#if !defined(PRESCAN_V0_NAIVE) && \
    !defined(PRESCAN_V1_WITH_SHARED_MEMORY) && \
    !defined(PRESCAN_V2_PADDED_BANK_CONFLICT) && \
    !defined(PRESCAN_V3_UNROLLED) && \
    !defined(PRESCAN_V4_BRANCHLESS) && \
    !defined(PRESCAN_V5_HILLIS_STEELE) && \
    !defined(PRESCAN_V6_RECURSIVE) && \
    !defined(PRESCAN_V7_BUFFER_REUSE) && \
    !defined(PRESCAN_V8_FOUR_ELEMS_PER_THREAD) && \
    !defined(PRESCAN_V9_THRUST) && \
    !defined(PRESCAN_V10_LAUNCH_BOUNDS) && \
    !defined(PRESCAN_V11_CACHE_CONFIG)
#define PRESCAN_V0_NAIVE
#endif

static inline int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

enum ScanKernelKind {
    KERNEL_BASE = 0,
    KERNEL_PADDED,
    KERNEL_UNROLLED,
    KERNEL_BRANCHLESS,
    KERNEL_HILLIS,
    KERNEL_LAUNCH_BOUNDS
};

__global__ void scanNaiveGlobal(float *outArray, const float *inArray, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < idx; ++i) {
        sum += inArray[i];
    }
    outArray[idx] = sum;
}

__global__ void scanBlockBase(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK];

    int threadId = threadIdx.x;
    int leftIndex = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    sharedMem[threadId] = (leftIndex < numElements) ? inArray[leftIndex] : 0.0f;
    sharedMem[threadId + BLOCK_SIZE] = (rightIndex < numElements) ? inArray[rightIndex] : 0.0f;

    __syncthreads();

    int stride = 1;
    while (stride < ELEMENTS_PER_BLOCK) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            sharedMem[index] += sharedMem[index - stride];
        }
        stride <<= 1;
        __syncthreads();
    }

    if (threadId == 0) {
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = sharedMem[ELEMENTS_PER_BLOCK - 1];
        }
        sharedMem[ELEMENTS_PER_BLOCK - 1] = 0.0f;
    }
    __syncthreads();

    stride = BLOCK_SIZE;
    while (stride > 0) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            float t = sharedMem[index - stride];
            sharedMem[index - stride] = sharedMem[index];
            sharedMem[index] += t;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (leftIndex < numElements) {
        outArray[leftIndex] = sharedMem[threadId];
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] = sharedMem[threadId + BLOCK_SIZE];
    }
}

__global__ void scanBlockPadded(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK + (ELEMENTS_PER_BLOCK / NUM_BANKS) + 1];

    int threadId = threadIdx.x;
    int leftIndex = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    int ai = threadId;
    int bi = threadId + BLOCK_SIZE;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    sharedMem[ai + bankOffsetA] = (leftIndex < numElements) ? inArray[leftIndex] : 0.0f;
    sharedMem[bi + bankOffsetB] = (rightIndex < numElements) ? inArray[rightIndex] : 0.0f;

    __syncthreads();

    int stride = 1;
    while (stride < ELEMENTS_PER_BLOCK) {
        int idx = (threadId + 1) * stride * 2 - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            int a = idx - stride;
            int b = idx;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            sharedMem[b] += sharedMem[a];
        }
        stride <<= 1;
        __syncthreads();
    }

    if (threadId == 0) {
        int last = ELEMENTS_PER_BLOCK - 1;
        int lastPadded = last + CONFLICT_FREE_OFFSET(last);
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = sharedMem[lastPadded];
        }
        sharedMem[lastPadded] = 0.0f;
    }
    __syncthreads();

    stride = BLOCK_SIZE;
    while (stride > 0) {
        int idx = (threadId + 1) * stride * 2 - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            int a = idx - stride;
            int b = idx;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            float t = sharedMem[a];
            sharedMem[a] = sharedMem[b];
            sharedMem[b] += t;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (leftIndex < numElements) {
        outArray[leftIndex] = sharedMem[ai + bankOffsetA];
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] = sharedMem[bi + bankOffsetB];
    }
}

__global__ void scanBlockUnrolled(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK];

    int threadId = threadIdx.x;
    int leftIndex = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    sharedMem[threadId] = (leftIndex < numElements) ? inArray[leftIndex] : 0.0f;
    sharedMem[threadId + BLOCK_SIZE] = (rightIndex < numElements) ? inArray[rightIndex] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int stride = 1; stride < ELEMENTS_PER_BLOCK; stride <<= 1) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            sharedMem[index] += sharedMem[index - stride];
        }
        __syncthreads();
    }

    if (threadId == 0) {
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = sharedMem[ELEMENTS_PER_BLOCK - 1];
        }
        sharedMem[ELEMENTS_PER_BLOCK - 1] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int stride = BLOCK_SIZE; stride > 0; stride >>= 1) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            float t = sharedMem[index - stride];
            sharedMem[index - stride] = sharedMem[index];
            sharedMem[index] += t;
        }
        __syncthreads();
    }

    if (leftIndex < numElements) {
        outArray[leftIndex] = sharedMem[threadId];
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] = sharedMem[threadId + BLOCK_SIZE];
    }
}

__global__ void scanBlockBranchless(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK];

    int threadId = threadIdx.x;
    int leftIndex = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    sharedMem[threadId] = (leftIndex < numElements) ? inArray[leftIndex] : 0.0f;
    sharedMem[threadId + BLOCK_SIZE] = (rightIndex < numElements) ? inArray[rightIndex] : 0.0f;

    __syncthreads();

    int stride = 1;
    while (stride < ELEMENTS_PER_BLOCK) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            sharedMem[index] += sharedMem[index - stride];
        }
        stride <<= 1;
        __syncthreads();
    }

    if (threadId == 0) {
        float blockTotal = sharedMem[ELEMENTS_PER_BLOCK - 1];
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = blockTotal;
        }
        sharedMem[ELEMENTS_PER_BLOCK - 1] = 0.0f;
    }
    __syncthreads();

    stride = BLOCK_SIZE;
    while (stride > 0) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            float leftVal = sharedMem[index - stride];
            float rightVal = sharedMem[index];
            sharedMem[index - stride] = rightVal;
            sharedMem[index] = rightVal + leftVal;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (leftIndex < numElements) {
        outArray[leftIndex] = sharedMem[threadId];
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] = sharedMem[threadId + BLOCK_SIZE];
    }
}

__global__ void scanBlockHillisSteele(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK];

    int t = threadIdx.x;
    int i0 = blockIdx.x * ELEMENTS_PER_BLOCK + t;
    int i1 = i0 + BLOCK_SIZE;

    sharedMem[t] = (i0 < numElements) ? inArray[i0] : 0.0f;
    sharedMem[t + BLOCK_SIZE] = (i1 < numElements) ? inArray[i1] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < ELEMENTS_PER_BLOCK; offset <<= 1) {
        float a0 = sharedMem[t];
        float a1 = sharedMem[t + BLOCK_SIZE];

        if (t >= offset) {
            a0 += sharedMem[t - offset];
        }
        if ((t + BLOCK_SIZE) >= offset) {
            a1 += sharedMem[t + BLOCK_SIZE - offset];
        }

        __syncthreads();
        sharedMem[t] = a0;
        sharedMem[t + BLOCK_SIZE] = a1;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float total = sharedMem[ELEMENTS_PER_BLOCK - 1];
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = total;
        }
    }

    float ex0 = (t == 0) ? 0.0f : sharedMem[t - 1];
    float ex1 = sharedMem[t + BLOCK_SIZE - 1];

    if (i0 < numElements) {
        outArray[i0] = ex0;
    }
    if (i1 < numElements) {
        outArray[i1] = ex1;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void scanBlockLaunchBounds(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK];

    int threadId = threadIdx.x;
    int leftIndex = blockIdx.x * ELEMENTS_PER_BLOCK + threadId;
    int rightIndex = leftIndex + BLOCK_SIZE;

    sharedMem[threadId] = (leftIndex < numElements) ? inArray[leftIndex] : 0.0f;
    sharedMem[threadId + BLOCK_SIZE] = (rightIndex < numElements) ? inArray[rightIndex] : 0.0f;

    __syncthreads();

    int stride = 1;
    while (stride < ELEMENTS_PER_BLOCK) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            sharedMem[index] += sharedMem[index - stride];
        }
        stride <<= 1;
        __syncthreads();
    }

    if (threadId == 0) {
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = sharedMem[ELEMENTS_PER_BLOCK - 1];
        }
        sharedMem[ELEMENTS_PER_BLOCK - 1] = 0.0f;
    }
    __syncthreads();

    stride = BLOCK_SIZE;
    while (stride > 0) {
        int index = (threadId + 1) * stride * 2 - 1;
        if (index < ELEMENTS_PER_BLOCK) {
            float t = sharedMem[index - stride];
            sharedMem[index - stride] = sharedMem[index];
            sharedMem[index] += t;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (leftIndex < numElements) {
        outArray[leftIndex] = sharedMem[threadId];
    }
    if (rightIndex < numElements) {
        outArray[rightIndex] = sharedMem[threadId + BLOCK_SIZE];
    }
}

__global__ void scanBlock4(float *outArray, const float *inArray, float *blockSums, int numElements)
{
    __shared__ float sharedMem[ELEMENTS_PER_BLOCK4];

    int t = threadIdx.x;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK4 + t;

    int i0 = base;
    int i1 = base + BLOCK_SIZE;
    int i2 = base + 2 * BLOCK_SIZE;
    int i3 = base + 3 * BLOCK_SIZE;

    sharedMem[t] = (i0 < numElements) ? inArray[i0] : 0.0f;
    sharedMem[t + BLOCK_SIZE] = (i1 < numElements) ? inArray[i1] : 0.0f;
    sharedMem[t + 2 * BLOCK_SIZE] = (i2 < numElements) ? inArray[i2] : 0.0f;
    sharedMem[t + 3 * BLOCK_SIZE] = (i3 < numElements) ? inArray[i3] : 0.0f;

    __syncthreads();

    int stride = 1;
    while (stride < ELEMENTS_PER_BLOCK4) {
        int idx = (t + 1) * stride * 2 - 1;
        if (idx < ELEMENTS_PER_BLOCK4) {
            sharedMem[idx] += sharedMem[idx - stride];
        }
        stride <<= 1;
        __syncthreads();
    }

    if (t == 0) {
        if (blockSums != NULL) {
            blockSums[blockIdx.x] = sharedMem[ELEMENTS_PER_BLOCK4 - 1];
        }
        sharedMem[ELEMENTS_PER_BLOCK4 - 1] = 0.0f;
    }
    __syncthreads();

    stride = ELEMENTS_PER_BLOCK4 >> 1;
    while (stride > 0) {
        int idx = (t + 1) * stride * 2 - 1;
        if (idx < ELEMENTS_PER_BLOCK4) {
            float tmp = sharedMem[idx - stride];
            sharedMem[idx - stride] = sharedMem[idx];
            sharedMem[idx] += tmp;
        }
        stride >>= 1;
        __syncthreads();
    }

    if (i0 < numElements) outArray[i0] = sharedMem[t];
    if (i1 < numElements) outArray[i1] = sharedMem[t + BLOCK_SIZE];
    if (i2 < numElements) outArray[i2] = sharedMem[t + 2 * BLOCK_SIZE];
    if (i3 < numElements) outArray[i3] = sharedMem[t + 3 * BLOCK_SIZE];
}

__global__ void addSumsBase(float *outArray, const float *blockSums, int numElements)
{
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

__global__ void addSums4(float *outArray, const float *blockSums, int numElements)
{
    int t = threadIdx.x;
    int base = blockIdx.x * ELEMENTS_PER_BLOCK4 + t;

    int i0 = base;
    int i1 = base + BLOCK_SIZE;
    int i2 = base + 2 * BLOCK_SIZE;
    int i3 = base + 3 * BLOCK_SIZE;

    float blockSum = blockSums[blockIdx.x];
    if (i0 < numElements) outArray[i0] += blockSum;
    if (i1 < numElements) outArray[i1] += blockSum;
    if (i2 < numElements) outArray[i2] += blockSum;
    if (i3 < numElements) outArray[i3] += blockSum;
}

static void launchScanKernel(ScanKernelKind kind,
                             float *outArray,
                             const float *inArray,
                             float *blockSums,
                             int numElements,
                             int numBlocks)
{
    switch (kind) {
        case KERNEL_BASE:
            scanBlockBase<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, blockSums, numElements);
            break;
        case KERNEL_PADDED:
            scanBlockPadded<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, blockSums, numElements);
            break;
        case KERNEL_UNROLLED:
            scanBlockUnrolled<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, blockSums, numElements);
            break;
        case KERNEL_BRANCHLESS:
            scanBlockBranchless<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, blockSums, numElements);
            break;
        case KERNEL_HILLIS:
            scanBlockHillisSteele<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, blockSums, numElements);
            break;
        case KERNEL_LAUNCH_BOUNDS:
            scanBlockLaunchBounds<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, blockSums, numElements);
            break;
    }
}

static void prescanRecursiveWithKind(float *outArray,
                                     const float *inArray,
                                     int numElements,
                                     ScanKernelKind kind)
{
    int numBlocks = divUp(numElements, ELEMENTS_PER_BLOCK);
    if (numBlocks <= 1) {
        launchScanKernel(kind, outArray, inArray, NULL, numElements, 1);
        return;
    }

    float *dev_block_sums = NULL;
    float *dev_block_prefix_sums = NULL;

    cudaMalloc((void**)&dev_block_sums, numBlocks * sizeof(float));
    cudaMalloc((void**)&dev_block_prefix_sums, numBlocks * sizeof(float));

    launchScanKernel(kind, outArray, inArray, dev_block_sums, numElements, numBlocks);
    prescanRecursiveWithKind(dev_block_prefix_sums, dev_block_sums, numBlocks, kind);
    addSumsBase<<<numBlocks, BLOCK_SIZE>>>(outArray, dev_block_prefix_sums, numElements);

    cudaFree(dev_block_sums);
    cudaFree(dev_block_prefix_sums);
}

static void prescanCurrentSharedMemory(float *outArray, float *inArray, int numElements)
{
    int numBlocks = divUp(numElements, ELEMENTS_PER_BLOCK);
    int numSubBlocks = divUp(numBlocks, ELEMENTS_PER_BLOCK);

    float *dev_block_sums = NULL;
    float *dev_block_prefix_sums = NULL;
    float *dev_sub_sums = NULL;
    float *dev_sub_prefix_sums = NULL;

    cudaMalloc((void**)&dev_block_sums, numBlocks * sizeof(float));
    cudaMalloc((void**)&dev_block_prefix_sums, numBlocks * sizeof(float));
    cudaMalloc((void**)&dev_sub_sums, numSubBlocks * sizeof(float));
    cudaMalloc((void**)&dev_sub_prefix_sums, numSubBlocks * sizeof(float));

    scanBlockBase<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, dev_block_sums, numElements);
    scanBlockBase<<<numSubBlocks, BLOCK_SIZE>>>(dev_block_prefix_sums, dev_block_sums, dev_sub_sums, numBlocks);
    scanBlockBase<<<1, BLOCK_SIZE>>>(dev_sub_prefix_sums, dev_sub_sums, NULL, numSubBlocks);

    addSumsBase<<<numSubBlocks, BLOCK_SIZE>>>(dev_block_prefix_sums, dev_sub_prefix_sums, numBlocks);
    addSumsBase<<<numBlocks, BLOCK_SIZE>>>(outArray, dev_block_prefix_sums, numElements);

    cudaFree(dev_block_sums);
    cudaFree(dev_block_prefix_sums);
    cudaFree(dev_sub_sums);
    cudaFree(dev_sub_prefix_sums);
}

static void prescanNaiveNoShared(float *outArray, float *inArray, int numElements)
{
    int threads = BLOCK_SIZE;
    int blocks = divUp(numElements, threads);
    scanNaiveGlobal<<<blocks, threads>>>(outArray, inArray, numElements);
}

static void prescanPadded(float *outArray, float *inArray, int numElements)
{
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_PADDED);
}

static void prescanUnrolled(float *outArray, float *inArray, int numElements)
{
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_UNROLLED);
}

static void prescanBranchless(float *outArray, float *inArray, int numElements)
{
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_BRANCHLESS);
}

static void prescanHillisSteele(float *outArray, float *inArray, int numElements)
{
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_HILLIS);
}

static void prescanRecursiveBaseline(float *outArray, float *inArray, int numElements)
{
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_BASE);
}

static float *g_reuse_block_sums = NULL;
static float *g_reuse_block_prefix_sums = NULL;
static int g_reuse_capacity = 0;

static void ensureReuseBuffers(int count)
{
    if (count <= g_reuse_capacity) {
        return;
    }
    if (g_reuse_block_sums) cudaFree(g_reuse_block_sums);
    if (g_reuse_block_prefix_sums) cudaFree(g_reuse_block_prefix_sums);

    cudaMalloc((void**)&g_reuse_block_sums, count * sizeof(float));
    cudaMalloc((void**)&g_reuse_block_prefix_sums, count * sizeof(float));
    g_reuse_capacity = count;
}

static void prescanBufferReuse(float *outArray, float *inArray, int numElements)
{
    int numBlocks = divUp(numElements, ELEMENTS_PER_BLOCK);
    ensureReuseBuffers(numBlocks);

    scanBlockBase<<<numBlocks, BLOCK_SIZE>>>(outArray, inArray, g_reuse_block_sums, numElements);
    prescanRecursiveWithKind(g_reuse_block_prefix_sums, g_reuse_block_sums, numBlocks, KERNEL_BASE);
    addSumsBase<<<numBlocks, BLOCK_SIZE>>>(outArray, g_reuse_block_prefix_sums, numElements);
}

static void prescanFourElemsPerThread(float *outArray, float *inArray, int numElements)
{
    int numBlocks4 = divUp(numElements, ELEMENTS_PER_BLOCK4);

    float *dev_block_sums = NULL;
    float *dev_block_prefix_sums = NULL;

    cudaMalloc((void**)&dev_block_sums, numBlocks4 * sizeof(float));
    cudaMalloc((void**)&dev_block_prefix_sums, numBlocks4 * sizeof(float));

    scanBlock4<<<numBlocks4, BLOCK_SIZE>>>(outArray, inArray, dev_block_sums, numElements);
    prescanRecursiveWithKind(dev_block_prefix_sums, dev_block_sums, numBlocks4, KERNEL_BASE);
    addSums4<<<numBlocks4, BLOCK_SIZE>>>(outArray, dev_block_prefix_sums, numElements);

    cudaFree(dev_block_sums);
    cudaFree(dev_block_prefix_sums);
}

static void prescanThrust(float *outArray, float *inArray, int numElements)
{
#if defined(PRESCAN_V9_THRUST)
    thrust::device_ptr<float> inPtr(inArray);
    thrust::device_ptr<float> outPtr(outArray);
    thrust::exclusive_scan(inPtr, inPtr + numElements, outPtr);
#else
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_BASE);
#endif
}

static void prescanLaunchBounds(float *outArray, float *inArray, int numElements)
{
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_LAUNCH_BOUNDS);
}

static void prescanCacheConfig(float *outArray, float *inArray, int numElements)
{
    cudaFuncSetCacheConfig(scanBlockBase, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(addSumsBase, cudaFuncCachePreferL1);
    prescanRecursiveWithKind(outArray, inArray, numElements, KERNEL_BASE);
}

#if defined(PRESCAN_V0_NAIVE)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanNaiveNoShared(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V1_WITH_SHARED_MEMORY)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanCurrentSharedMemory(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V2_PADDED_BANK_CONFLICT)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanPadded(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V3_UNROLLED)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanUnrolled(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V4_BRANCHLESS)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanBranchless(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V5_HILLIS_STEELE)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanHillisSteele(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V6_RECURSIVE)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanRecursiveBaseline(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V7_BUFFER_REUSE)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanBufferReuse(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V8_FOUR_ELEMS_PER_THREAD)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanFourElemsPerThread(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V9_THRUST)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanThrust(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V10_LAUNCH_BOUNDS)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanLaunchBounds(outArray, inArray, numElements);
}
#elif defined(PRESCAN_V11_CACHE_CONFIG)
void prescanArray(float *outArray, float *inArray, int numElements)
{
    if (numElements <= 0) return;
    prescanCacheConfig(outArray, inArray, numElements);
}
#endif

#endif // _PRESCAN_CU_
