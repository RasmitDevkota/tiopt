# Define directories for dependencies
SHTOOLS_LIB_DIR += /usr/local/lib
MOD_DIR += /usr/local/include
FFTW_LIB_DIR = /usr/lib
GMSH_INCLUDE_DIR = $(GMSH_DIR)/api
GMSH_LIB_DIR = /usr/local/lib
NLOPT_LIB_DIR = /usr/local/lib
SPARSELIZARD_INCLUDE_DIR = $(SPARSELIZARD_DIR)/src
SPARSELIZARD_LIB_DIR = /usr/local/lib

# Compiler and flags
FC = gfortran
CC = gcc
CXX = g++

CFLAGS = -O2 -Wall -fPIC
CXXFLAGS = -Wall

FFTW_LIBS = -L$(FFTW_LIB_DIR) -lfftw3
MATH_LIBS = -lm
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS
SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
GMSH_LIBS = -L$(GMSH_LIB_DIR) -lgmsh
NLOPT_LIBS = -L$(NLOPT_LIB_DIR) -lnlopt
SPARSELIZARD_LIBS = -L$(SPARSELIZARD_LIB_DIR) -lsparselizard -lopenblas -L$(SLEPC_DIR)/$(SLEPC_ARCH)/lib -lslepc -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

SPARSELIZARD_INCLUDES = -I$(SPARSELIZARD_INCLUDE_DIR)
SPARSELIZARD_INCLUDES += $(shell find $(SPARSELIZARD_INCLUDE_DIR) -type d -exec echo -I{} \;)
SPARSELIZARD_INCLUDES += -I$(SLEPC_DIR)/$(SLEPC_ARCH)/include -I$(SLEPC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include

# Source files (C)
C_SRC = data_structures.c ti_utils.c verlet.c sph.c electrostatics.c trap_geometry.c main.c

# Source files (C++)
CXX_SRC = fem_wrapper.cpp

# Object files
C_OBJ = data_structures.o ti_utils.o verlet.o sph.o electrostatics.o trap_geometry.o main.o

# Object files (C++)
CXX_OBJ = fem_wrapper.o

# Output binary
OUTPUT = tiopt

# Make target: all
all: $(OUTPUT)

# Link the final executable
$(OUTPUT): $(C_OBJ) $(CXX_OBJ)
	$(CXX) $(C_OBJ) $(CXX_OBJ) $(GMSH_LIBS) $(SPARSELIZARD_LIBS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(NLOPT_LIBS) -o $(OUTPUT)

# Compile C code
$(C_OBJ): $(C_SRC)
	$(CC) -c $(C_SRC) -I$(GMSH_INCLUDE_DIR) $(GMSH_LIBS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(NLOPT_LIBS) $(CFLAGS)

# Compile C++ code
$(CXX_OBJ): $(CXX_SRC)
	$(CXX) -c $(CXX_SRC) $(CXXFLAGS) -I$(GMSH_INCLUDE_DIR) $(GMSH_LIBS) $(SPARSELIZARD_INCLUDES) $(SPARSELIZARD_LIBS) -o $(CXX_OBJ)

# Clean up build files
clean:
	rm -f $(CXX_OBJ) $(C_OBJ) $(OUTPUT)

# Phony targets
.PHONY: all clean

