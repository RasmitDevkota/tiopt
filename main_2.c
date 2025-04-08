#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern void compute_shcoeffs(double *grid, int *nlat, int *nlon, int *lmax, double *alm);

#define PI14 3.14159265358979
#define NLAT 40
#define NLON 40
#define LMAX 20

int main() {
    double grid[NLAT][NLON];
    double alm[(LMAX+1)*(LMAX+1)*2]; // Flattened 1D array to store real and imaginary parts

    int nlat = NLAT, nlon = NLON, lmax = LMAX;

    // Fill grid with sample data
    for (int i = 0; i < NLAT; i++) {
        for (int j = 0; j < NLON; j++) {
            grid[i][j] = 0.0;

            // grid[i][j] += 1.0;
            // grid[i][j] += 1 / sqrt((float)4 * PI14);

            // grid[i][j] += sqrt((float) 3 / 2) * sin((double)i / NLAT * PI14);
            // grid[i][j] += sqrt((float) 3) * cos((double)i / NLAT * PI14);
            grid[i][j] += cos((double)i / NLAT * PI14);

            // grid[i][j] += cos((double)i / NLAT * PI14) * (cos((double)i / NLAT * PI14) + sin((double)i / NLAT * PI14));

            // grid[i][j] += sin((double)i / NLAT * PI14) * cos((double)i / NLAT * PI14);
            // grid[i][j] += sin((double)i / NLON * PI14) * cos((double)i / NLON * PI14);
            // grid[i][j] += sin((double)i / NLON * PI14) * cos((double)i / NLON * PI14);
        }
    }

    // Call the Fortran function
    compute_shcoeffs((double *)grid, &nlat, &nlon, &lmax, alm);

    // Print first few coefficients
    for (int l = 0; l <= LMAX; l++) {
        for (int m = -l; m <= l; m++) {
            if (m < 0)
            {
                double alm_real = pow(-1, m) * alm[2 * (l * (LMAX + 1) + m)];
                double alm_imag = pow(-1, m) * alm[2 * (l * (LMAX + 1) + m) + 1];
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

    return 0;
}

