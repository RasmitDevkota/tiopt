# Define directories for dependencies
GSL_LIB_DIR = /usr/lib
FFTW_LIB_DIR = /usr/lib
SHTOOLS_LIB_DIR += /usr/local/lib
MOD_DIR += /usr/local/include
NLOPT_LIB_DIR = /usr/local/lib

# Library under test
TIOPT_DIR = ..

MATH_LIBS = -lm
GSL_LIBS = -lopenblas -L$(GSL_LIB_DIR) -lgsl
FFTW_LIBS = -L$(FFTW_LIB_DIR) -lfftw3
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS
SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
NLOPT_LIBS = -L$(NLOPT_LIB_DIR) -lnlopt
TIOPT_LIBS = -L$(TIOPT_DIR) -ltiopt -Wl,-rpath,$(abspath $(TIOPT_DIR)) -Wl,-rpath,$(SHTOOLS_LIB_DIR)

# Compiler and flags
FC = gfortran
CC = gcc
CXX = g++

CFLAGS = -O1 -Wall -fPIC -g
CXXFLAGS = -O1 -Wall -fPIC -g

# Source files (C)
C_SRC = test_utils.c test_ti_utils.c test_electrodynamics.c test_io.c test_verlet.c test_waveforms.c test_photonics.c test_quantum.c test_transport.c test_analysis.c test_acceleration.c test_codesign.c main.c

# Header files (C)
C_HDR = $(wildcard *.h) $(wildcard $(TIOPT_DIR)/*.h)

# Object files
C_OBJ = $(C_SRC:.c=.o)

# Output binary
OUTPUT = tiopt-test

# Make target: all
all: $(OUTPUT)

# Link the final executable
$(OUTPUT): $(C_OBJ) $(TIOPT_DIR)/libtiopt.so
	$(CC) $(C_OBJ) $(TIOPT_LIBS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(GSL_LIBS) $(NLOPT_LIBS) -o $(OUTPUT)

# Compile C code
%.o: %.c $(C_HDR)
	$(CC) -c $< $(CFLAGS) -I$(TIOPT_DIR) -o $@

# Build the library under test
$(TIOPT_DIR)/libtiopt.so:
	$(MAKE) -C $(TIOPT_DIR) libtiopt.so

# Run the tests
test: $(OUTPUT)
	./$(OUTPUT)

# Benchmark the library on the CPU (serially and in parallel) and on the OpenCL device, if there is one
BENCHMARK = tiopt-benchmark

$(BENCHMARK): benchmark.o $(TIOPT_DIR)/libtiopt.so
	$(CC) benchmark.o $(TIOPT_LIBS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(GSL_LIBS) $(NLOPT_LIBS) -o $(BENCHMARK)

benchmark: $(BENCHMARK)
	./$(BENCHMARK)

# Clean up build files
clean:
	rm -f $(C_OBJ) $(OUTPUT) benchmark.o $(BENCHMARK)

# Deep clean build files and outputs
deep-clean:
	rm -f $(C_OBJ) $(OUTPUT) benchmark.o $(BENCHMARK)
	rm meshes/*

# Phony targets
.PHONY: all clean deep-clean test benchmark
