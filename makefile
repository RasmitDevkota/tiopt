# Define directories for dependencies
GSL_LIB_DIR = /usr/lib
FFTW_LIB_DIR = /usr/lib
SHTOOLS_LIB_DIR += /usr/local/lib
MOD_DIR += /usr/local/include
NLOPT_LIB_DIR = /usr/local/lib

MATH_LIBS = -lm
GSL_LIBS = -lopenblas -L$(GSL_LIB_DIR) -lgsl
FFTW_LIBS = -L$(FFTW_LIB_DIR) -lfftw3_threads -lfftw3
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS
SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
NLOPT_LIBS = -L$(NLOPT_LIB_DIR) -lnlopt

# Multithreading with OpenMP (part of GCC), which can be disabled with make OPENMP=0
OPENMP ?= 1
ifeq ($(OPENMP), 1)
OPENMP_FLAGS = -fopenmp
endif

# OpenCL is loaded at runtime when a GPU is requested, which only needs the dynamic loader
DL_LIBS = -ldl

# Also search the dependency directories at runtime, so that libtiopt.so can be loaded from Python without LD_LIBRARY_PATH
RPATH_FLAGS = -Wl,-rpath,$(SHTOOLS_LIB_DIR) -Wl,-rpath,$(NLOPT_LIB_DIR)

LIBS = $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(FFTW_LIBS) $(MATH_LIBS) $(GSL_LIBS) $(NLOPT_LIBS) $(DL_LIBS) $(OPENMP_FLAGS) $(RPATH_FLAGS)

# Compiler and flags
FC = gfortran
CC = gcc
CXX = g++

# Compile-time parameters from defs.h can be overridden, e.g. make DEFINES="-DNSPH_X=32" for a longer region of
# spherical harmonics expansions (run make clean first so that everything is rebuilt)
DEFINES ?=

CFLAGS = -O3 -Wall -fPIC -g -fno-math-errno -fcx-limited-range $(OPENMP_FLAGS) $(DEFINES)
CXXFLAGS = -O3 -Wall -fPIC -g $(OPENMP_FLAGS) $(DEFINES)

# Source files (C)
LIB_SRC = ti_utils.c experiments.c verlet.c electrodynamics.c io.c transport.c waveforms.c analysis.c traps.c ions.c photonics.c quantum.c circuits.c codesign.c acceleration.c bindings.c ion_heating/ion_heating.c
C_SRC = $(LIB_SRC) main.c

# Header files (C)
C_HDR = $(wildcard *.h) ion_heating/ion_heating.h

# Object files
LIB_OBJ = $(LIB_SRC:.c=.o)
C_OBJ = $(LIB_OBJ) main.o

# Output binary
OUTPUT = tiopt

# Output shared library, used by the Python bindings, GUI and tests
LIBRARY = libtiopt.so

# Make target: all
all: $(OUTPUT) $(LIBRARY)

# Link the final executable
$(OUTPUT): $(C_OBJ)
	$(CC) $(C_OBJ) $(LIBS) -o $(OUTPUT)

# Link the shared library
$(LIBRARY): $(LIB_OBJ)
	$(CC) -shared $(LIB_OBJ) $(LIBS) -o $(LIBRARY)

# Compile C code
%.o: %.c $(C_HDR)
	$(CC) -c $< $(CFLAGS) -o $@

# Embed the OpenCL kernels in the library as a string
kernels_cl.h: kernels.cl
	echo "static const char kernels_cl[] = {" > kernels_cl.h
	od -An -v -tx1 kernels.cl | sed 's/\([0-9a-f][0-9a-f]\)/0x\1,/g' >> kernels_cl.h
	echo "0x00 };" >> kernels_cl.h

acceleration.o: kernels_cl.h

# Run the C and Python test suites
test: $(LIBRARY)
	$(MAKE) -C tests test
	PYTHONPATH=python python3 -m pytest tests

# Time the main computations with different numbers of threads and devices
benchmark: $(LIBRARY)
	$(MAKE) -C tests benchmark

# Launch the GUI
gui: $(LIBRARY)
	PYTHONPATH=python python3 -m tiopt

# Clean up build files
clean:
	rm -f $(C_OBJ) $(OUTPUT) $(LIBRARY) kernels_cl.h
	$(MAKE) -C tests clean

# Deep clean build files and outputs
deep-clean:
	rm -f $(C_OBJ) $(OUTPUT) $(LIBRARY) kernels_cl.h
	$(MAKE) -C tests clean
	rm meshes/*

# Phony targets
.PHONY: all clean deep-clean test gui benchmark
