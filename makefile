# Define directories for dependencies
SHTOOLS_LIB_DIR = /usr/local/lib
MOD_DIR = /usr/local/include
FFTW_LIB_DIR = /usr/lib

# Compiler and flags
FC = gfortran
CC = gcc

CFLAGS = -O2 -Wall -fPIC

FFTW_LIBS = -lfftw3
MATH_LIBS = -lm

SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS

# Source files
C_SRC = main.c verlet.c sph.c ti_utils.c

# Object files
C_OBJ = main.o verlet.o sph.o ti_utils.o

# Output binary
OUTPUT = tiopt

# Make target: all
all: $(OUTPUT)

# Link the final executable
$(OUTPUT): $(C_OBJ)
	$(FC) $(C_OBJ) $(SHWRAPPER_LIBS) -L$(FFTW_LIB_DIR) $(FFTW_LIBS) $(MATH_LIBS) -o $(OUTPUT)

# Compile C code
$(C_OBJ): $(C_SRC)
	$(CC) -c $(C_SRC) $(CFLAGS)

# Clean up build files
clean:
	rm -f $(C_OBJ) $(OUTPUT)

# Phony targets
.PHONY: all clean

