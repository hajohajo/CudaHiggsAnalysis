IDIR=`root-config --incdir`
CFLAGS=-I$(IDIR)
NVCC=nvcc


all: main.cpp Reader_CPU.cpp TriggerSelection.o METFilterSelection.o TauSelection.o wrapper.o
	gcc -std=c++11 -I `root-config --incdir` -o main main.cpp Reader_CPU.cpp TriggerSelection.o METFilterSelection.o TauSelection.o wrapper.o `root-config --libs` -L/usr/local/cuda-9.0/lib64 -lcudart -lstdc++ -lboost_filesystem -lboost_system

TauSelection.o: CudaSelections/TauSelection.cu
	$(NVCC) -c CudaSelections/TauSelection.cu

wrapper.o: CudaSelections/wrapper.cu
	$(NVCC) -c CudaSelections/wrapper.cu

TriggerSelection.o: CudaSelections/TriggerSelection.cu
	$(NVCC) -c CudaSelections/TriggerSelection.cu

METFilterSelection.o: CudaSelections/METFilterSelection.cu
	$(NVCC) -c CudaSelections/METFilterSelection.cu
