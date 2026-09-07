# tiopt

tiopt is a C/Fortran-based program which simulates ion traps and ion transport, with the goal of optimizing trap design, transport waveforms, and ion control. It aims to enable all of the following in one tool:
- Construct a seed trap design from presets, consisting of electrodes and (optionally) integrated photonics and electronics
- Simulate trap components (e.g. electrodes, waveguide unit cells)
- Represent electric potentials of electrodes using spherical harmonic expansions
- Obtain curve fits of parameters of photonic components
- Simulate ion transport via robust symplectic integration
- Simulate ion-photon interactions during laser pulses
- Analyze various aspects of ion control based on experiment data
- Optimize all aspects of trap geometry, transport waveforms, and ion addressing based on analysis

# Installation

## Unix-based systems

1. Install the required dependencies based on your distribution (this is based on Ubuntu)
    - Git
    - gcc (this is our c compiler)
    - make
    - gfortran
    - libgsl-dev
    - libfftw3-dev
    - libnlopt-dev
    - libblas-dev
    - liblapack-dev
    - libopenblas-dev
2. Clone repositories
    - Make a new folder and clone the following repositories into that folder
    ```
    git clone https://github.com/SHTOOLS/SHTOOLS.git
    git clone https://github.com/RasmitDevkota/shtools_wrapper.git
    git clone https://github.com/RasmitDevkota/shtools_wrapper.git
    ```
3. Build and install SHTOOLS
    - Enter the SHTOOLS directory in the terminal
    - Build the Fortran version
    ```
    make fortran
    ```
    - Install SHTOOLS
    ```
    sudo make install
    ```
4. Build shtools_wrapper
    - Enter the shtools_wrapper directory in the terminal
    - Open the file called "makefile" in shtools_wrapper using any code editor
    - Change
    ```
    SHTOOLS_SRC = /home/rasmitdevkota/projects/SHTOOLS/src
    SHTOOLS_LIB = /home/rasmitdevkota/projects/SHTOOLS
    ```
    to
    ```
    SHTOOLS_SRC = /usr/local/include
    SHTOOLS_LIB = /usr/local/lib
    ```
    - Save the file
    - Build the wrapper
    ```
    sudo make
    ```
5. Build Tiopt
    - Enter the tiopt directory
    - To clean any previous build files, run
    ```
    make clean
    ```
    - Build tiopt
    ```
    make
    ```
    - Now, there should be an executable file called tiopt.
6. Run Tiopt
    - Run the executable
    ```
    ./tiopt
    ```

