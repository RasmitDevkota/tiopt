#include "sph.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void compute_shcoeffs(double *grid, int *nlat, int *nlon, int *lmax, double *alm);

#define NLAT 40
#define NLON 40
#define LMAX 20

void expand_spherical_harmonics(
    const int f_len,
    double (*f)[f_len]
) {
    double grid[NLAT][NLON];
    double alm[(LMAX+1)*(LMAX+1)*2]; // @TODO - store somewhere instead

    int nlat = NLAT, nlon = NLON, lmax = LMAX;

    for (int i = 0; i < NLAT; i++) {
        for (int j = 0; j < NLON; j++) {
            grid[i][j] = 0.0; // @TODO - perform 3D interpolation to populate sampling grid
        }
    }

    // Call the Fortran function
    compute_shcoeffs((double*) grid, &nlat, &nlon, &lmax, alm);

    // Print first few coefficients
    for (int l = 0; l <= LMAX; l++) {
        for (int m = 0; m <= l; m++) {
            printf("a_lm[%d,%d] = (%f, %f)\n", l, m, alm[2 * (l * (LMAX + 1) + m)],   // Real part
                                               alm[2 * (l * (LMAX + 1) + m) + 1]); // Imaginary part
        }
    }
}
