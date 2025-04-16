# Define directories for dependencies
SHTOOLS_LIB_DIR = /usr/local/lib
MOD_DIR = /usr/local/include
FFTW_LIB_DIR = /usr/lib
GMSH_INCLUDE_DIR = /home/rasmitdevkota/projects/comp-libs/gmsh/api
GMSH_LIB_DIR = /usr/local/lib
NLOPT_LIB_DIR = /usr/local/lib

# Compiler and flags
FC = gfortran
CC = gcc
CXX = g++

CFLAGS = -O2 -Wall -fPIC
CXXFLAGS = -Wall

FFTW_LIBS = -L$(FFTW_LIB_DIR) -lfftw3
MATH_LIBS = -lm
SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS
GMSH_LIBS = -L$(GMSH_LIB_DIR) -lgmsh
NLOPT_LIBS = -L$(NLOPT_LIB_DIR) -lnlopt

# Source files (C)
# C_SRC = main.c ti_utils.c trap_geometry.c sph.c verlet.c
C_SRC = ti_utils.c verlet.c trap_geometry.c main.c

# Source files (C++)
# CXX_SRC = electrostatics.cpp

# Object files
# C_OBJ = main.o ti_utils.o trap_geometry.o sph.o verlet.o
C_OBJ = ti_utils.o verlet.o trap_geometry.o main.o

# Object files (C++)
# CXX_OBJ = electrostatics.o

# Output binary
OUTPUT = tiopt

# Make target: all
all: $(OUTPUT)

# Link the final executable
$(OUTPUT): $(C_OBJ) $(CXX_OBJ)
	$(CC) $(C_OBJ) $(CXX_OBJ) $(MATH_LIBS) $(NLOPT_LIBS) -o $(OUTPUT)

# Compile C code
$(C_OBJ): $(C_SRC)
	$(CC) -c $(C_SRC) -I$(GMSH_INCLUDE_DIR) $(GMSH_LIBS) $(MATH_LIBS) $(NLOPT_LIBS) $(CFLAGS)

# Compile C++ code
$(CXX_OBJ): $(CXX_SRC)
	$(CXX) -c $(CXX_SRC) $(CXXFLAGS) -o $(CXX_OBJ)

# Clean up build files
clean:
	rm -f $(CXX_OBJ) $(C_OBJ) $(OUTPUT)

# Phony targets
.PHONY: all clean

