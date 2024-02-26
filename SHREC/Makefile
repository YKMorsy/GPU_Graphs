# Compiler
CC = g++

# Compiler flags
# CFLAGS = -Wall -Wextra -std=c++20

# Source files
SRCS = include/csr/csr.cpp include/cpu/cpuBFS.cpp main.cpp

# Object files (substitute .cpp with .o)
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = graphBFS

# Default rule
all: $(EXEC)

# Rule to build executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC)

# Rule to compile source files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(EXEC)

# Rule to run the program with input
run: $(EXEC)
	./$(EXEC) $(ARGS)
	rm -f $(OBJS) $(EXEC)

.PHONY: all clean run
