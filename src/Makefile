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
GEN-EXE:=gen-circles


ALL: $(EXE) $(OMP-EXE) $(CUDA-EXE) $(GEN-EXE)

$(EXE).movie: CFLAGS+=-DMOVIE
$(EXE).movie: $(EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

movie: $(EXE).movie
	./$(EXE).movie 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi

$(OMP-EXE): CFLAGS+=$(OMP-FLAGS)
$(OMP-EXE): $(OMP-EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# omp-dbg: CFLAGS+=-DDBG $(OMP-FLAGS)
# omp-dbg: $(OMP-EXE).c
# 	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

# serial-dbg: CFLAGS+=-DDBG
# serial-dbg: $(EXE).c
# 	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)
# 
# omp-test: serial-dbg omp-dbg $(GEN-EXE)	
# 	for i in $$(seq 1 1) ; do \
# 		./$(GEN-EXE) 10000; \
# 		./serial-dbg > res_serial_time.txt ; \
# 		./omp-dbg > res_omp_time.txt ; \
# 		sed 's/([^)]*)//' res_serial_time.txt | head -n -1 > res_serial.txt ; \
# 		sed 's/([^)]*)//' res_omp_time.txt | head -n -1 > res_omp.txt ; \
# 		if ! cmp -s res_serial.txt res_omp.txt ; then \
# 			echo "Errore: I file di output per l'iterazione $${i} sono diversi." ; \
# 		exit 1; \
# 		fi; \
# 		echo "Iterazione $${i} OK"; \
# 	done

clean:
	\rm -f $(EXE) $(EXE).movie $(OMP-EXE) $(CUDA-EXE) $(GEN-EXE) serial-dbg omp-dbg cuda-dbg *.txt *.o *~ *.gp *.png *.avi 