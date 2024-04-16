#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "me.h"
#include "tables.h"

struct mv_data
{
  int sad;
  int mv_x;
  int mv_y;
};

__device__ static void sad_block_8x8(uint8_t *block1, uint8_t *block2, int stride, int *result)
{
  int u, v;

  *result = 0;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      *result += abs(block2[v*stride+u] - block1[v*stride+u]);
    }
  }
}

/* Motion estimation for 8x8 block */
__global__ static void me_block_8x8(struct c63_common *cm, struct macroblock *mb_gpu, uint8_t *orig, uint8_t *ref, int color_component)
{
  int mb_x = blockIdx.x;
  int mb_y = blockIdx.y;
  struct macroblock *mb = &mb_gpu[mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  int x = mb_x * 8 + threadIdx.x - range;
  int y = mb_y * 8 + threadIdx.y - range;

  // Store all SADs in a flat array such that we can find the minimun SAD later
  extern __shared__ struct mv_data sad_array[];
  
  int flattenedThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;
  
  int mx = mb_x * 8;
  int my = mb_y * 8;
  int sad;

  sad_array[flattenedThreadIdx].mv_x = x - mx;
  sad_array[flattenedThreadIdx].mv_y = y - my;

  if (x < 0 || y < 0 || x > w - 8 || y > h - 8) {
    sad_array[flattenedThreadIdx].sad = INT_MAX;
  } else {
    sad_block_8x8(orig + my*w+mx, ref + y*w+x, w, &sad);

    // Store the SAD for this thread in the appropriate index in the array
    sad_array[flattenedThreadIdx].sad = sad;
  }

  __syncthreads();

  // Sequential addressing minimum algorithm
  for(int stride = (blockDim.x*blockDim.y)/2; stride > 1; stride /= 2)
  {
    // Gives out way too high values
    // Each iteration the amount of threads working will be halved since we compare 2 elements each iteration
    if(flattenedThreadIdx < stride)
    {
      if(sad_array[flattenedThreadIdx].sad > sad_array[flattenedThreadIdx + stride].sad)
      {
        sad_array[flattenedThreadIdx] = sad_array[flattenedThreadIdx + stride];
        // printf("sad_array[%d] = %d\n", flattenedThreadIdx, sad_array[flattenedThreadIdx])
      }
    }

    __syncthreads();
  }

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    mb->mv_x = sad_array[0].mv_x;
    mb->mv_y = sad_array[0].mv_y;
    mb->use_mv = 1;
  }
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  struct c63_common *cm_gpu;
  struct macroblock *mb_Y, *mb_U, *mb_V;

  cudaMalloc((void **)&cm_gpu, sizeof(struct c63_common));

  cudaMalloc((void **)&mb_Y, sizeof(struct macroblock)*(cm->mb_rows)*(cm->mb_cols));
  cudaMalloc((void **)&mb_U, sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2));
  cudaMalloc((void **)&mb_V, sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2));

  cudaMemcpy(cm_gpu, cm, sizeof(struct c63_common), cudaMemcpyHostToDevice);

  cudaMemcpy(mb_Y, cm->curframe->mbs[Y_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows)*(cm->mb_cols), cudaMemcpyHostToDevice);
  cudaMemcpy(mb_U, cm->curframe->mbs[U_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2), cudaMemcpyHostToDevice);
  cudaMemcpy(mb_V, cm->curframe->mbs[V_COMPONENT], sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2), cudaMemcpyHostToDevice);

  uint8_t *orig_Y, *recons_Y;
  cudaMalloc((void **)&orig_Y, sizeof(uint8_t)*cm->ypw*cm->yph);
  cudaMalloc((void **)&recons_Y, sizeof(uint8_t)*cm->ypw*cm->yph);

  uint8_t *orig_U, *recons_U;
  cudaMalloc((void **)&orig_U, sizeof(uint8_t)*cm->upw*cm->uph);
  cudaMalloc((void **)&recons_U, sizeof(uint8_t)*cm->upw*cm->uph);

  uint8_t *orig_V, *recons_V;
  cudaMalloc((void **)&orig_V, sizeof(uint8_t)*cm->vpw*cm->vph);
  cudaMalloc((void **)&recons_V, sizeof(uint8_t)*cm->vpw*cm->vph);
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy(orig_Y, cm->curframe->orig->Y, sizeof(uint8_t)*cm->ypw*cm->yph, cudaMemcpyHostToDevice);
  cudaMemcpy(orig_U, cm->curframe->orig->U, sizeof(uint8_t)*cm->upw*cm->uph, cudaMemcpyHostToDevice);
  cudaMemcpy(orig_V, cm->curframe->orig->V, sizeof(uint8_t)*cm->vpw*cm->vph, cudaMemcpyHostToDevice);
  cudaMemcpy(recons_Y, cm->curframe->recons->Y, sizeof(uint8_t)*cm->ypw*cm->yph, cudaMemcpyHostToDevice);
  cudaMemcpy(recons_U, cm->curframe->recons->U, sizeof(uint8_t)*cm->upw*cm->uph, cudaMemcpyHostToDevice);
  cudaMemcpy(recons_V, cm->curframe->recons->V, sizeof(uint8_t)*cm->vpw*cm->vph, cudaMemcpyHostToDevice);

  dim3 lumaThreadsPerBlock(cm->me_search_range*2, cm->me_search_range*2);
  dim3 lumaGridDim(cm->mb_cols ,cm->mb_rows);

  /* Luma */
  me_block_8x8<<<lumaGridDim, lumaThreadsPerBlock, lumaThreadsPerBlock.x*lumaThreadsPerBlock.y*sizeof(int)*3>>>(cm_gpu, mb_Y, orig_Y, recons_Y, Y_COMPONENT);

  dim3 chromaThreadsPerBlock(cm->me_search_range, cm->me_search_range);
  dim3 chromaGridDim(cm->mb_cols/2, cm->mb_rows/2);

  /* Chroma */
  me_block_8x8<<<chromaGridDim, chromaThreadsPerBlock, chromaThreadsPerBlock.x*chromaThreadsPerBlock.y*sizeof(int)*3>>>(cm_gpu, mb_U, orig_U, recons_U, U_COMPONENT);
  me_block_8x8<<<chromaGridDim, chromaThreadsPerBlock, chromaThreadsPerBlock.x*chromaThreadsPerBlock.y*sizeof(int)*3>>>(cm_gpu, mb_V, orig_V, recons_V, V_COMPONENT);

  cudaDeviceSynchronize();

  cudaMemcpy(cm->curframe->mbs[Y_COMPONENT], mb_Y, sizeof(struct macroblock)*(cm->mb_rows)*(cm->mb_cols), cudaMemcpyDeviceToHost);
  cudaMemcpy(cm->curframe->mbs[U_COMPONENT], mb_U, sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2), cudaMemcpyDeviceToHost);
  cudaMemcpy(cm->curframe->mbs[V_COMPONENT], mb_V, sizeof(struct macroblock)*(cm->mb_rows/2)*(cm->mb_cols/2), cudaMemcpyDeviceToHost);

  cudaFree(orig_Y);
  cudaFree(recons_Y);
  cudaFree(orig_U);
  cudaFree(recons_U);
  cudaFree(orig_V);
  cudaFree(recons_V);
  cudaFree(cm_gpu);
  cudaFree(mb_Y);
  cudaFree(mb_U);
  cudaFree(mb_V);
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
