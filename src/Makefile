# Available targets:
#
# - make
#   builds both the omp and the cuda executables
#
# - make clean
#   remove all output files and executables
#
# - make omp-circles
#   compile the executable omp-circles.c
#
# - cuda-circles
#   compile the executable cuda-circles.cu

CFLAGS=-std=c99 -Wall -Wpedantic
LDLIBS+=-lm
OMP-FLAGS:=-fopenmp
OMP-EXE:=omp-circles
CUDA-EXE:=cuda-circles

ALL: $(OMP-EXE) $(CUDA-EXE)

$(OMP-EXE): CFLAGS+=$(OMP-FLAGS)
$(OMP-EXE): $(OMP-EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(CUDA-EXE): $(CUDA-EXE).cu
	nvcc $< -o $@

clean:
	\rm -f $(OMP-EXE) $(CUDA-EXE) *.txt *.o *~