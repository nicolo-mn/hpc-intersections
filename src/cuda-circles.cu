/*************************************
 * Nicolò Monaldini matr: 0001031164 
 *************************************/

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
        /* each thread computes the displacements of one circle */
        for (int j=0; j<ncircles; j++) {
            const float deltax = circles->x[j] - circles->x[gindex];
            const float deltay = circles->y[j] - circles->y[gindex];
            const float dist = hypotf(deltax, deltay);
            const float Rsum = circles->r[gindex] + circles->r[j];
            if (dist < Rsum - EPSILON && j != gindex) {
                if (j > gindex)
                    n_intersections[lindex]++;
                const float overlap = Rsum - dist;
                assert(overlap > 0.0);
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
 * Create and populate the struct of array `circles` with randomly placed
 * circles.
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
    int n_overlaps;
    int *d_n_overlaps;
    circle_t *d_circles;
    float *d_circles_dx, *d_circles_dy, *d_circles_x, *d_circles_y, *d_circles_r;
    /* allocations */
    cudaSafeCall( cudaMalloc((void**)&d_n_overlaps, sizeof(*d_n_overlaps)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles, sizeof(*d_circles)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_x, n*sizeof(*d_circles_x)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_y, n*sizeof(*d_circles_y)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_r, n*sizeof(*d_circles_r)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_dx, n*sizeof(*d_circles_dx)) );
    cudaSafeCall( cudaMalloc((void**)&d_circles_dy, n*sizeof(*d_circles_dy)) );
    /* copies */
    cudaSafeCall( cudaMemcpy(d_circles_x, circles->x, n*sizeof(*circles->x), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_y, circles->y, n*sizeof(*circles->y), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_r, circles->r, n*sizeof(*circles->r), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_dx, circles->dx, n*sizeof(*circles->dx), cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMemcpy(d_circles_dy, circles->dy, n*sizeof(*circles->dy), cudaMemcpyHostToDevice) );
    /* struct's pointers binding */
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
