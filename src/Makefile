project: main.o timing.o kernels.o
	nvcc -arch=sm_86 -o project main.o timing.o kernels.o -I.

main.o: main.cu
	nvcc -arch=sm_86 -c main.cu

kernels.o: kernels.cu kernels.h
	nvcc -arch=sm_86 -c kernels.cu

timing.o: timing.c timing.h
	gcc -c -x c timing.c -l.

clean:
	rm -r *.o project