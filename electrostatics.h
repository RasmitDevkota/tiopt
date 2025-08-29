#include <stdio.h>

// #include <sparselizard.h>

#include "defs.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern void compute_shcoeffs_cmplx(double *grid, int *nlat, int *nlon, int *lmax, double *alm);
extern void compute_shcoeffs_real(double *grid, int *nlat, int *nlon, int *lmax, double *alm);

void sparselizard_wrapper(
    char *mesh_filename,
    char *output_filename,
    double (*Vlm)[(LMAX+1)*(LMAX+1)*2]
);

void sparselizard_sample_dh1(
    void *_f,
    int sample_volume,
    double grid[NLAT][NLON]
);

void expand_spherical_harmonics(
    double grid[NLAT][NLON],
    double (*alm)[(LMAX+1)*(LMAX+1)*2]
);

#ifdef __cplusplus
}
#endif

