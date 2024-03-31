# Compiler
CC = g++
NVCC = nvcc
DPCPP = icpx -fsycl

# Source files
SRCS = include/csr/csr.cpp include/cpu/cpuBFS.cpp include/gpu_sycl/syclBFS.cpp main_sycl.cpp

# Object files (substitute .cpp and .cu with .o)
OBJS = $(filter %.o, $(SRCS:.cpp=.o))

# Executable name
EXEC = graphBFS

# Default rule
all: $(EXEC)

# Rule to build executable
$(EXEC): $(OBJS)
	@$(DPCPP) $(OBJS) -o $(EXEC)

# Rule to compile source files (.cpp)
%.o: %.cpp
	@$(DPCPP) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	@rm -f $(OBJS) $(EXEC)

# Rule to run the program with input
run: $(EXEC)
	@./$(EXEC) $(ARGS)
	@rm -f $(OBJS) $(EXEC)

.PHONY: all clean run
