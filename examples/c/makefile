# Build the C example against libtiopt in the root of the repository
TIOPT_DIR = ../..
SHTOOLS_LIB_DIR = /usr/local/lib

CC = gcc
CFLAGS = -O2 -Wall -I$(TIOPT_DIR)
LIBS = -L$(TIOPT_DIR) -ltiopt -Wl,-rpath,$(abspath $(TIOPT_DIR)) -Wl,-rpath,$(SHTOOLS_LIB_DIR) -lm

all: basic_trap

basic_trap: basic_trap.c $(TIOPT_DIR)/libtiopt.so
	$(CC) $(CFLAGS) basic_trap.c $(LIBS) -o basic_trap

$(TIOPT_DIR)/libtiopt.so:
	$(MAKE) -C $(TIOPT_DIR) libtiopt.so

# Run from this directory, since the example reads the example trap and circuit by relative paths
run: basic_trap
	./basic_trap

clean:
	rm -f basic_trap

.PHONY: all run clean
