# Define directories for dependencies
SHTOOLS_LIB_DIR += /usr/local/lib
MOD_DIR += /usr/local/include
FFTW_LIB_DIR = /usr/lib
GMSH_INCLUDE_DIR = $(GMSH_DIR)/api
GMSH_LIB_DIR = /usr/local/lib
NLOPT_LIB_DIR = /usr/local/lib
SPARSELIZARD_INCLUDE_DIR = $(SPARSELIZARD_DIR)/src
SPARSELIZARD_LIB_DIR = /usr/local/lib

FFTW_LIBS = -L$(FFTW_LIB_DIR) -lfftw3
MATH_LIBS = -lm
SHTOOLS_LIBS = -L$(SHTOOLS_LIB_DIR) -lSHTOOLS
SHWRAPPER_LIBS = -L$(SHTOOLS_LIB_DIR) -lwrapper
GMSH_LIBS = -L$(GMSH_LIB_DIR) -lgmsh
NLOPT_LIBS = -L$(NLOPT_LIB_DIR) -lnlopt
SPARSELIZARD_LIBS = -L$(SPARSELIZARD_LIB_DIR) -lsparselizard -lopenblas -L$(SLEPC_DIR)/$(SLEPC_ARCH)/lib -lslepc -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc

# GMSH_INCLUDES = -I$(GMSH_INCLUDE_DIR) # dependency errors flood output
GMSH_INCLUDES = -isystem $(GMSH_INCLUDE_DIR)

# SPARSELIZARD_INCLUDES = -I$(SPARSELIZARD_INCLUDE_DIR) # dependency errors flood output
SPARSELIZARD_INCLUDES = -isystem $(SPARSELIZARD_INCLUDE_DIR)
SPARSELIZARD_INCLUDES += $(shell find $(SPARSELIZARD_INCLUDE_DIR) -type d -exec echo -I{} \;)
# SPARSELIZARD_INCLUDES += -I$(SLEPC_DIR)/$(SLEPC_ARCH)/include -I$(SLEPC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(PETSC_DIR)/include # dependency errors flood output
SPARSELIZARD_INCLUDES += -isystem $(SLEPC_DIR)/$(SLEPC_ARCH)/include -isystem $(SLEPC_DIR)/include -isystem $(PETSC_DIR)/$(PETSC_ARCH)/include -isystem $(PETSC_DIR)/include

# Compiler and flags
FC = gfortran
CC = gcc
CXX = g++

CFLAGS = -O1 -Wall -fPIC -g
CXXFLAGS = -O1 -Wall -fPIC -g

# Source files (C)
C_SRC = ti_utils.c verlet.c electrostatics_wrapper.c trap_geometry.c io.c main.c

# Source files (C++)
CXX_SRC = electrostatics.cpp

# Object files
C_OBJ = ti_utils.o verlet.o electrostatics_wrapper.o trap_geometry.o io.o main.o

# Object files (C++)
CXX_OBJ = electrostatics.o

# Output binary
OUTPUT = tiopt

# Make target: all
all: $(OUTPUT)

# Link the final executable
$(OUTPUT): $(C_OBJ) $(CXX_OBJ)
	$(CXX) $(C_OBJ) $(CXX_OBJ) $(GMSH_LIBS) $(SPARSELIZARD_LIBS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(NLOPT_LIBS) -o $(OUTPUT)

# Compile C code
$(C_OBJ): $(C_SRC)
	$(CC) -c $(C_SRC) $(CFLAGS) $(GMSH_INCLUDES) $(GMSH_LIBS) $(SPARSELIZARD_INCLUDES) $(SPARSELIZARD_LIBS) $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(MATH_LIBS) $(NLOPT_LIBS)

# Compile C++ code
$(CXX_OBJ): $(CXX_SRC)
	$(CXX) -c $(CXX_SRC) $(CXXFLAGS) $(GMSH_INCLUDES) $(GMSH_LIBS) $(SPARSELIZARD_INCLUDES) $(SPARSELIZARD_LIBS) -o $(CXX_OBJ)

# Clean up build files
clean:
	rm -f $(CXX_OBJ) $(C_OBJ) $(OUTPUT)

# Deep clean build files and outputs
deep-clean:
	rm -f $(CXX_OBJ) $(C_OBJ) $(OUTPUT)
	rm meshes/*

# Phony targets
.PHONY: all clean

