#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* Constants */
const float XMIN = 0.0;
const float XMAX = 1000.0;
const float YMIN = 0.0;
const float YMAX = 1000.0;
const float RMIN = 10.0;
const float RMAX = 100.0;
const char FILENAME[] = "init.txt";

/* Function to generate random float in [a, b] */
float randab(float a, float b)
{
    return a + (((float)rand()) / RAND_MAX) * (b - a);
}

int main( int argc, char* argv[] ) {

    int n;
    float cx, cy, cr;

    if (argc != 2) {
        fprintf(stderr, "ERRORE: numero di argomenti errato in gen.c.\n");
        return EXIT_FAILURE;
    }
    n = atoi(argv[1]);

    FILE *file = fopen(FILENAME, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        cx = randab(XMIN, XMAX);
        cy = randab(YMIN, YMAX);
        cr = randab(RMIN, RMAX);

        // Write coordinates to the file
        fprintf(file, "%f %f %f\n", cx, cy, cr);
    }

    fclose(file);
    return 0;
}
