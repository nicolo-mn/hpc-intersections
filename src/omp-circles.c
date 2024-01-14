/*************************************
 * Nicolò Monaldini matr: 0001031164 
 *************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

typedef struct {
    float x, y;   /* coordinates of center */
    float r;      /* radius */
} circle_t;

#define XMIN 0.0f
#define XMAX 1000.0f
#define YMIN 0.0f
#define YMAX 1000.0f
#define RMIN 10.0f
#define RMAX 100.0f
#define EPSILON 1e-5f
#define K 1.5f

int ncircles;
circle_t *circles = NULL;
/* displacements due to interactions with other circles */
float* circles_dx = NULL;
float* circles_dy = NULL;

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
    ncircles = n;
    circles = (circle_t*)malloc(n * sizeof(*circles));
    assert(circles != NULL);
    circles_dx = (float*)malloc(n * sizeof(*circles_dx));
    assert(circles_dx != NULL);
    circles_dy = (float*)malloc(n * sizeof(*circles_dy));
    assert(circles_dy != NULL);
    for (int i=0; i<n; i++) {
        circles[i].x = randab(XMIN, XMAX);
        circles[i].y = randab(YMIN, YMAX);
        circles[i].r = randab(RMIN, RMAX);
        circles_dx[i] = circles_dy[i] = 0.0;
    }
}

/**
 * Compute the force acting on each circle; returns the number of
 * overlapping pairs of circles (each overlapping pair must be counted
 * only once).
 */
int compute_forces( void )
{
    int n_intersections = 0;
    #pragma omp parallel for default(none) shared(circles, ncircles) \
    reduction(+:n_intersections) \
    reduction(+:circles_dx[:ncircles]) \
    reduction(+:circles_dy[:ncircles]) \
    schedule(dynamic, 32)
    for (int i=0; i<ncircles; i++) {
        for (int j=i+1; j<ncircles; j++) {
            const float deltax = circles[j].x - circles[i].x;
            const float deltay = circles[j].y - circles[i].y;
            const float dist = hypotf(deltax, deltay);
            const float Rsum = circles[i].r + circles[j].r;
            if (dist < Rsum - EPSILON) {
                n_intersections++;
                const float overlap = Rsum - dist;
                assert(overlap > 0.0);
                // avoid division by zero
                const float overlap_x = overlap / (dist + EPSILON) * deltax;
                const float overlap_y = overlap / (dist + EPSILON) * deltay;
                circles_dx[i] -= overlap_x / K;
                circles_dy[i] -= overlap_y / K;
                circles_dx[j] += overlap_x / K;
                circles_dy[j] += overlap_y / K;
            }
        }
    }
    return n_intersections;
}

/**
 * Updates the position and the resets the displacements of each circle.
 */
void update_circles( void ) {
    #pragma omp parallel for default(none) shared(circles, circles_dx, circles_dy, ncircles)
    for (int i=0; i<ncircles; i++) {
        circles[i].x += circles_dx[i];
        circles[i].y += circles_dy[i];
        circles_dx[i] = circles_dy[i] = 0.0;
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
void dump_circles( int iterno )
{
    char fname[64];
    snprintf(fname, sizeof(fname), "circles-%05d.gp", iterno);
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
        fprintf(out, "%f %f %f\n", circles[i].x, circles[i].y, circles[i].r);
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
    dump_circles(0);
#endif
    for (int it=0; it<iterations; it++) {
        const double tstart_iter = hpc_gettime();
        const int n_overlaps = compute_forces();
        update_circles();
        const double elapsed_iter = hpc_gettime() - tstart_iter;
#ifdef MOVIE
        dump_circles(it+1);
#endif
        printf("Iteration %d of %d, %d overlaps (%f s)\n", it+1, iterations, n_overlaps, elapsed_iter);
    }
    const double elapsed_prog = hpc_gettime() - tstart_prog;
    printf("Elapsed time: %f\n", elapsed_prog);

    free(circles);
    free(circles_dx);
    free(circles_dy);

    return EXIT_SUCCESS;
}
