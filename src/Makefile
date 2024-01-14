# Default makefile for the circles intersections program.
#
# Available targets:
#
# - make
#   builds the standard executable
#
# - make clean
#   remove all output files and executables
#
# - make movie
#   compile the executable circles.movie that writes a gnuplot
#   file at each iteration; the executable is run and the output
#   is processed to produce an animation circles.avi

# Last modified 2023-12-06 by Moreno Marzolla

EXE:=circles
CFLAGS=-std=c99 -Wall -Wpedantic
LDLIBS+=-lm
OMP-FLAGS:=-fopenmp
OMP-EXE:=omp-circles
CUDA-EXE:=cuda-circles

ALL: $(EXE) $(OMP-EXE) $(CUDA-EXE)

$(EXE).movie: CFLAGS+=-DMOVIE
$(EXE).movie: $(EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

movie: $(EXE).movie
	./$(EXE).movie 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi

$(OMP-EXE).movie: CFLAGS+=-DMOVIE $(OMP-FLAGS)
$(OMP-EXE).movie: $(OMP-EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

omp-movie: $(OMP-EXE).movie
	./$(OMP-EXE).movie 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 omp-circles.avi

$(CUDA-EXE).movie: $(CUDA-EXE).cu
	nvcc -DMOVIE $< -o $@ 

cuda-movie: $(CUDA-EXE).movie
	./$(CUDA-EXE).movie 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 cuda-circles.avi

$(OMP-EXE): CFLAGS+=$(OMP-FLAGS)
$(OMP-EXE): $(OMP-EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(CUDA-EXE): $(CUDA-EXE).cu
	nvcc $< -o $@

clean:
	\rm -f $(EXE) $(EXE).movie $(OMP-EXE) $(OMP-EXE).movie $(CUDA-EXE) $(CUDA-EXE).movie *.txt *.o *~ *.gp *.png *.avi

# CFLAGS=-std=c99 -Wall -Wpedantic
# LDLIBS+=-lm
# OMP-FLAGS:=-fopenmp
# OMP-EXE:=omp-circles
# CUDA-EXE:=cuda-circles

# ALL: $(OMP-EXE) $(CUDA-EXE)

# $(OMP-EXE): CFLAGS+=$(OMP-FLAGS)
# $(OMP-EXE): $(OMP-EXE).c
# 	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# $(CUDA-EXE): $(CUDA-EXE).cu
# 	nvcc $< -o $@

# clean:
# 	\rm -f $(OMP-EXE) $(CUDA-EXE) *.txt *.o *~ *.gp *.png *.avi