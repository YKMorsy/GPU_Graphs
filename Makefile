# Compiler
CC = g++
NVCC = nvcc

# Compiler flags
# CFLAGS = -Wall -Wextra -std=c++20
# NVCCFLAGS = -std=c++20

# Source files
SRCS = include/csr/csr.cpp include/cpu/cpuBFS.cpp include/gpu/gpuBFS.cu main.cu

# Object files (substitute .cpp and .cu with .o)
OBJS = $(filter %.o, $(SRCS:.cpp=.o) $(SRCS:.cu=.o))

# Executable name
EXEC = graphBFS

# Default rule
all: $(EXEC)

# Rule to build executable
$(EXEC): $(OBJS)
	@$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(EXEC)

# Rule to compile source files (.cpp)
%.o: %.cpp
	@$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile CUDA source files (.cu)
%.o: %.cu
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	@rm -f $(OBJS) $(EXEC)

# Rule to run the program with input
run: $(EXEC)
	@./$(EXEC) $(ARGS)
	@rm -f $(OBJS) $(EXEC)

.PHONY: all clean run
