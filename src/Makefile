project: main.o timing.o kernels.o cpu_kernels.o
	nvcc -arch=sm_86 -o project main.o timing.o kernels.o cpu_kernels.o -I.

main.o: main.cu
	nvcc -arch=sm_86 -c main.cu

kernels.o: kernels.cu kernels.h
	nvcc -arch=sm_86 -c kernels.cu

cpu_kernels.o: cpu_kernels.c cpu_kernels.h
	g++ -c cpu_kernels.c -o cpu_kernels.o

timing.o: timing.c timing.h
	gcc -c -x c timing.c -o timing.o

clean:
	rm -f *.o project