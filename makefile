# Define directories for dependencies
GSL_LIB_DIR = /usr/lib
FFTW_LIB_DIR = /usr/lib
SHTOOLS_LIB_DIR += /usr/local/lib
MOD_DIR += /usr/local/include
NLOPT_LIB_DIR = /usr/local/lib

MATH_LIBS = -lm
GSL_LIBS = -lopenblas -L$(GSL_LIB_DIR) -lgsl
FFTW_LIBS = -L$(FFTW_LIB_DIR) -lfftw3
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS
SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
NLOPT_LIBS = -L$(NLOPT_LIB_DIR) -lnlopt

# Compiler and flags
FC = gfortran
CC = gcc
CXX = g++

CFLAGS = -O1 -Wall -fPIC -g
CXXFLAGS = -O1 -Wall -fPIC -g

# Source files (C)
C_SRC = ti_utils.c experiments.c verlet.c electrodynamics.c io.c main.c

# Object files
C_OBJ = ti_utils.o experiments.o verlet.o electrodynamics.o io.o main.o

# Output binary
OUTPUT = tiopt

# Make target: all
all: $(OUTPUT)

# Link the final executable
$(OUTPUT): $(C_OBJ)
	$(CC) $(C_OBJ) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(GSL_LIBS) $(NLOPT_LIBS) -o $(OUTPUT)

# Compile C code
$(C_OBJ): $(C_SRC)
	$(CC) -c $(C_SRC) $(CFLAGS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(GSL_LIBS) $(NLOPT_LIBS)

# Clean up build files
clean:
	rm -f $(C_OBJ) $(OUTPUT)

# Deep clean build files and outputs
deep-clean:
	rm -f $(C_OBJ) $(OUTPUT)
	rm meshes/*

# Phony targets
.PHONY: all clean

