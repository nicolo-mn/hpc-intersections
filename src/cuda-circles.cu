/****************************************************************************
 *
 * circles.c - Circles intersection
 *
 * Copyright (C) 2023 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ****************************************************************************/

/***
% Circles intersection
% Moreno Marzolla <moreno.marzolla@unibo.it>
% Last updated on 2023-12-06

This is a serial implementation of the circle intersection program
described in the specification.

To compile:

        gcc -std=c99 -Wall -Wpedantic circles.c -o circles -lm

To execute:

        ./circles [ncircles [iterations]]

where `ncircles` is the number of circles, and `iterations` is the
number of iterations to execute.

If you want to produce a movie (this is not required, and should be
avoided when measuring the performance of the parallel versions of
this program) compile with:

        gcc -std=c99 -Wall -Wpedantic -DMOVIE circles.c -o circles.movie -lm

and execute with:

        ./circles.movie 200 500

A lot of `circles-xxxxx.gp` files will be produced; these files must
be processed using `gnuplot` to create individual frames:

        for f in *.gp; do gnuplot "$f"; done

and then assembled to produce the movie `circles.avi`:

        ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi

***/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

typedef struct {
    float *x, *y;   /* coordinates of center */
    float *r;       /* radius */
    float *dx, *dy; /* displacements due to interactions with other circles */
} circle_t;

/* These constants can be replaced with #define's if necessary */
#define BLKDIM 1024
#define XMIN 0.0f
#define XMAX 1000.0f
#define YMIN 0.0f
#define YMAX 1000.0f
#define RMIN 10.0f
#define RMAX 100.0f
#define EPSILON 1e-5f
#define K 1.5f

circle_t *circles = NULL;

/**
 * Compute the force acting on each circle; returns the number of
 * overlapping pairs of circles (each overlapping pair must be counted
 * only once).
 */
__global__ void compute_forces( circle_t *circles, int *result, int ncircles ) {
    __shared__ int n_intersections[BLKDIM];
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    const int lindex = threadIdx.x;
    float my_dx = 0.0;
    float my_dy = 0.0;
    int bsize = blockDim.x / 2;

    if (gindex == 0) {
        *result = 0;
    }

    if (gindex < ncircles) {
        n_intersections[lindex] = 0;
        for (int j=0; j<ncircles; j++) {
            const float deltax = circles->x[j] - circles->x[gindex];
            const float deltay = circles->y[j] - circles->y[gindex];
            /* hypotf(x,y) computes sqrtf(x*x + y*y) avoiding
               overflow. This function is defined in <math.h>, and
               should be available also on CUDA. In case of troubles,
               it is ok to use sqrtf(x*x + y*y) instead. */
            const float dist = hypotf(deltax, deltay);
            const float Rsum = circles->r[gindex] + circles->r[j];
            if (dist < Rsum - EPSILON && j != gindex) {
                if (j > gindex)
                    n_intersections[lindex]++;
                const float overlap = Rsum - dist;
                assert(overlap > 0.0);
                // avoid division by zero
                const float overlap_x = overlap / (dist + EPSILON) * deltax;
                const float overlap_y = overlap / (dist + EPSILON) * deltay;
                my_dx -= overlap_x / K;
                my_dy -= overlap_y / K;
            }
        }

        circles->dx[gindex] = my_dx;
        circles->dy[gindex] = my_dy;

        /* wait for all threads to finish the copy operation */
        __syncthreads(); 

        /* All threads within the block cooperate to compute the local sum */
        while ( bsize > 0 ) {
            if ( lindex < bsize && gindex + bsize < ncircles) {
                n_intersections[lindex] += n_intersections[lindex + bsize];
            }
            bsize = bsize / 2; 
            /* threads must synchronize before performing the next
            reduction step */
            __syncthreads(); 
        }

        if ( 0 == lindex ) {
            atomicAdd(result, n_intersections[0]);
        }
    }
}

/**
 * Updates the position and the resets the displacements of each circle.
 */
__global__ void update_circles( circle_t *circles, int ncircles ) {
    const int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    if (gindex < ncircles) {
        circles->x[gindex] += circles->dx[gindex];
        circles->y[gindex] += circles->dy[gindex];
        circles->dx[gindex] = circles->dy[gindex] = 0.0;
    }
}

/**
 * Return a random float in [a, b]
 */
float randab(float a, float b)
{
    return a + (((float)rand())/RAND_MAX) * (b-a);
}

/**
 * Create and populate the array `circles[]` with randomly placed
 * circls.
 *
 * Do NOT parallelize this function.
 */
void init_circles(int n)
{
    assert(circles == NULL);
    circles = (circle_t*)malloc(sizeof(*circles));  assert(circles != NULL);
    circles->x = (float*)malloc(n*sizeof(*circles->x)); assert(circles->x != NULL);
    circles->y = (float*)malloc(n*sizeof(*circles->y)); assert(circles->y != NULL);
    circles->r = (float*)malloc(n*sizeof(*circles->r)); assert(circles->r != NULL);
    circles->dx = (float*)malloc(n*sizeof(*circles->dx)); assert(circles->dx != NULL);
    circles->dy = (float*)malloc(n*sizeof(*circles->dy)); assert(circles->dy != NULL);
    for (int i=0; i<n; i++) {
        circles->x[i] = randab(XMIN, XMAX);
        circles->y[i] = randab(YMIN, YMAX);
        circles->r[i] = randab(RMIN, RMAX);
        circles->dx[i] = circles->dy[i] = 0.0;
    }
}

#ifdef MOVIE
/**
 * Dumps the circles into a text file that can be processed using
 * gnuplot. This function may be used for debugging purposes, or to
 * produce a movie of how the algorithm works.
 *
 * You may want to completely remove this function from the final
 * version.
 */
void dump_circles( int iterno, int ncircles )
{
    char fname[64];
    snprintf(fname, sizeof(fname), "circles-cuda-%05d.gp", iterno);
    FILE *out = fopen(fname, "w");
    const float WIDTH = XMAX - XMIN;
    const float HEIGHT = YMAX - YMIN;
    fprintf(out, "set term png notransparent large\n");
    fprintf(out, "set output \"circles-%05d.png\"\n", iterno);
    fprintf(out, "set xrange [%f:%f]\n", XMIN - WIDTH*.2, XMAX + WIDTH*.2 );
    fprintf(out, "set yrange [%f:%f]\n", YMIN - HEIGHT*.2, YMAX + HEIGHT*.2 );
    fprintf(out, "set size square\n");
    fprintf(out, "plot '-' with circles notitle\n");
    for (int i=0; i<ncircles; i++) {
        fprintf(out, "%f %f %f\n", circles->x[i], circles->y[i], circles->r[i]);
    }
    fprintf(out, "e\n");
    fclose(out);
}
#endif

int main( int argc, char* argv[] )
{
    int n = 10000;
    int iterations = 20;

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [ncircles [iterations]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (argc > 2) {
        iterations = atoi(argv[2]);
    }

    init_circles(n);
    const double tstart_prog = hpc_gettime();
#ifdef MOVIE
    dump_circles(0, n);
#endif
    int n_overlaps;
    int *d_n_overlaps;
    circle_t *d_circles;
    float *d_circles_dx, *d_circles_dy, *d_circles_x, *d_circles_y, *d_circles_r;
    // allocations
    cudaSafeCall( cudaMalloc((void**)&d_n_overlaps, sizeof(*d_n_overlaps)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles, sizeof(*d_circles)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_x, n*sizeof(*d_circles_x)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_y, n*sizeof(*d_circles_y)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_r, n*sizeof(*d_circles_r)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_dx, n*sizeof(*d_circles_dx)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_dy, n*sizeof(*d_circles_dy)) );
    // copies
    cudaSafeCall( cudaMemcpy(d_circles_x, circles->x, n*sizeof(*circles->x), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_y, circles->y, n*sizeof(*circles->y), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_r, circles->r, n*sizeof(*circles->r), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_dx, circles->dx, n*sizeof(*circles->dx), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_dy, circles->dy, n*sizeof(*circles->dy), cudaMemcpyHostToDevice) );
    // struct's pointers binding
    cudaSafeCall( cudaMemcpy(&d_circles->x, &d_circles_x, sizeof(d_circles_x), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(&d_circles->y, &d_circles_y, sizeof(d_circles_y), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(&d_circles->r, &d_circles_r, sizeof(d_circles_r), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(&d_circles->dx, &d_circles_dx, sizeof(d_circles_dx), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(&d_circles->dy, &d_circles_dy, sizeof(d_circles_dy), cudaMemcpyHostToDevice) );
    for (int it=0; it<iterations; it++) {
        const double tstart_iter = hpc_gettime();
        compute_forces<<<(n+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_circles, d_n_overlaps, n);
        cudaCheckError();      
        cudaSafeCall( cudaMemcpy(&n_overlaps, d_n_overlaps, sizeof(n_overlaps), cudaMemcpyDeviceToHost) );
        update_circles<<<(n+BLKDIM-1)/BLKDIM, BLKDIM>>>(d_circles, n);
        cudaCheckError();
        const double elapsed_iter = hpc_gettime() - tstart_iter;
#ifdef MOVIE
        cudaSafeCall( cudaMemcpy(circles->x, d_circles_x, n*sizeof(*circles->x), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(circles->y, d_circles_y, n*sizeof(*circles->y), cudaMemcpyDeviceToHost) );
        cudaSafeCall( cudaMemcpy(circles->r, d_circles_r, n*sizeof(*circles->r), cudaMemcpyDeviceToHost) );
        dump_circles(it+1, n);
#endif
        printf("Iteration %d of %d, %d overlaps (%f s)\n", it+1, iterations, n_overlaps, elapsed_iter);
    }
    const double elapsed_prog = hpc_gettime() - tstart_prog;
    printf("Elapsed time: %f\n", elapsed_prog);

    cudaSafeCall( cudaFree(d_n_overlaps) );
    cudaSafeCall( cudaFree(d_circles_x) );
    cudaSafeCall( cudaFree(d_circles_y) );
    cudaSafeCall( cudaFree(d_circles_r) );
    cudaSafeCall( cudaFree(d_circles_dx) );
    cudaSafeCall( cudaFree(d_circles_dy) );
    cudaSafeCall( cudaFree(d_circles) );
    free(circles->x);
    free(circles->y);
    free(circles->r);
    free(circles->dx);
    free(circles->dy);
    free(circles);

    return EXIT_SUCCESS;
}
