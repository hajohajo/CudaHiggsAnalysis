IDIR=`root-config --incdir`
CFLAGS=-I$(IDIR)
NVCC=nvcc


all: main.cpp Reader.cpp TriggerSelection.o TauSelection.o wrapper.o
	gcc -std=c++11 -I `root-config --incdir` -o main main.cpp Reader.cpp TriggerSelection.o TauSelection.o wrapper.o -lstdc++ `root-config --libs` -L/usr/local/cuda-9.0/lib64 -lcudart

TauSelection.o: CudaSelections/TauSelection.cu
	$(NVCC) -c CudaSelections/TauSelection.cu

wrapper.o: CudaSelections/wrapper.cu
	$(NVCC) -c CudaSelections/wrapper.cu

TriggerSelection.o: CudaSelections/TriggerSelection.cu
	$(NVCC) -c CudaSelections/TriggerSelection.cu
