#include "sph.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void compute_shcoeffs_cmplx(double *grid, int *nlat, int *nlon, int *lmax, double *alm);
extern void compute_shcoeffs_real(double *grid, int *nlat, int *nlon, int *lmax, double *alm);

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
    // compute_shcoeffs_cmplx((double*) grid, &nlat, &nlon, &lmax, alm);
    compute_shcoeffs_real((double*) grid, &nlat, &nlon, &lmax, alm);

    // Print first few coefficients
    for (int l = 0; l <= LMAX; l++) {
        for (int m = -l; m <= l; m++) {
            if (m < 0)
            {
                // Y_l,-m = (-1)^m Y_l,m*
                double alm_real = pow(-1, m) * alm[2 * (l * (LMAX + 1) + m)];
                double alm_imag = pow(-1, m) * -1 * alm[2 * (l * (LMAX + 1) + m) + 1];
                printf("a_lm[%d,%d] = (%f, %f)\n", l, m, alm_real, alm_imag);
            }
            else
            {
                double alm_real = alm[2 * (l * (LMAX + 1) + m)];
                double alm_imag = alm[2 * (l * (LMAX + 1) + m) + 1];
                printf("a_lm[%d,+%d] = (%f, %f)\n", l, m, alm_real, alm_imag);
            }
        }
    }
}
