/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    // shared memory tiles for M and N
    const int tile_width = 16; // get tile width from blk dim
    __shared__ float M_shared[tile_width][tile_width];
    __shared__ float N_shared[tile_width][tile_width];

    // get row and col, each thread comps one element in p
    // tile indicies + which row/ col inside the tile
    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    // each block = 16 * 16 tile,
    // each thread = one element in the tile

    float sum = 0;
    int num_tiles = (M.width + tile_width - 1) / tile_width;

    for (int t = 0; t < num_tiles; t++){
        // get the element of M (cols) and N (rows) to load into shared memory
        int col_m = t * tile_width + threadIdx.x;
        int row_n = t * tile_width + threadIdx.y;

        // now load M into SM
        if (row < M.height && col_m < M.width)
            // if in bounds
            M_shared[threadIdx.y][threadIdx.x] = M.elements[row * M.width + col_m];
        else
            // out of bounds, set to 0
            M_shared[threadIdx.y][threadIdx.x] = 0.0;
        
        // now load N into SM
        if (col < N.width && row_n < N.height)
            // if in bounds
            N_shared[threadIdx.y][threadIdx.x] = N.elements[row_n * N.width + col];
        else
            // out of bounds, set to 0
            N_shared[threadIdx.y][threadIdx.x] = 0.0;
        
        // all threads in tile should be synced (loaded into shared mem)
        __syncthreads();

        for (int i = 0; i < tile_width; i++){
            // get partial sum for this tile, then sync 
            sum += M_shared[threadIdx.y][i] * N_shared[i][threadIdx.x];
        }
        __syncthreads();
    }   
    if(row < P.height && col < P.width)
        P.elements[row * P.width + col] = sum;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
