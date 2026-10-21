# Define directories for dependencies
GSL_LIB_DIR = /usr/lib
FFTW_LIB_DIR = /usr/lib
SHTOOLS_LIB_DIR += /usr/local/lib
MOD_DIR += /usr/local/include
NLOPT_LIB_DIR = /usr/local/lib

SRCDIR:= src
BUILDDIR := build
INCDIR:=include

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

CFLAGS = -O1 -Wall -fPIC -g -I$(INCDIR)
CXXFLAGS = -O1 -Wall -fPIC -g

# Source files (C)
C_SRC = ti_utils.c experiments.c verlet.c electrodynamics.c io.c main.c transport.c

# Object files
C_OBJ = ti_utils.o experiments.o verlet.o electrodynamics.o io.o main.o transport.o

# Output binary
OUTPUT = tiopt

# every .c file in SRCDIR, minus unfinished files listed in EXCLUDE
EXCLUDE  = $(SRCDIR)/analysis.c $(SRCDIR)/waveforms.c 
SRC      = $(filter-out $(EXCLUDE),$(wildcard $(SRCDIR)/*.c))
OBJECTS  = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRC))

# Make target: all
all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)	#we want to make BIN, so we need all the OBJECTS
	@echo "linking and producing the final application" 	#the @ at the start of line says that we don't see the command being executed, but it still is. not functionally impacted
	@$(CC) $^ $(SHTOOLS_LIBS) $(SHWRAPPER_LIBS) $(GSL_LIBS) $(NLOPT_LIBS) $(MATH_LIBS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	@echo "Compiling the $@ file"
	@$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: $(OUTPUT)
	./tiopt

#Clean up build files
clean:
	rm -rf $(BUILDDIR) $(OUTPUT)

# Deep clean build files and outputs
deep-clean: clean
	rm -f meshes/*

# Phony target
.PHONY: all clean deep-clean

