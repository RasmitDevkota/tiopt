#include "electrodynamics.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "ti_utils.h"

void solve_trap_electrodynamics
(
	struct Trap *trap,
	enum ElectrodynamicsSolver electrodynamics_solver
)
{
	// @TODO - implement parallelism
	for (int e = 0; e < trap->n_electrodes; e++)
	{
		printf("Solving electrodynamics for electrode %d/%d\n", e+1, trap->n_electrodes);

		struct Electrode electrode = (*trap->electrodes)[e];

		if (electrodynamics_solver == RELAXATION)
		{
			double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ] = calloc(RELAXATION_NX * RELAXATION_NY * RELAXATION_NZ, sizeof(double));

			int dx = 1;
			int dy = 1;
			int dz = 1;

			solver_relaxation(
				&electrode,
				V,
				dx, dy, dz,
				(int) 1E3,
				0
			);

			printf("Writing electric potential to CSV format...\n");
			FILE *f = fopen("V.txt", "w");
			for (int x = 0; x < RELAXATION_NX; x++)
				for (int y = 0; y < RELAXATION_NY; y++)
					for (int z = 0; z < +RELAXATION_NZ; z++)
						if ((*V)[x][y][z] > 0.0 || (*V)[x][y][z] < 0.0)
							fprintf(f, "%d,%d,%d,%f\n", x, y, z, (*V)[x][y][z]);
			fclose(f);

			printf("Sampling grid and expanding potential in spherical harmonics basis...\n");
			for (int sx = 0; sx < NSPH_X; sx++)
				for (int sy = 0; sy < NSPH_Y; sy++)
					for (int sz = 0; sz < NSPH_Z; sz++)
					{
						double x_c = sx * SPH_SPACING + SPH_R;
						double y_c = sy * SPH_SPACING + SPH_R;
						double z_c = sz * SPH_SPACING + SPH_Z_MIN + 1; // # @TEST - check the +1
						printf("Sphere at index (%d, %d, %d), position (%f, %f, %f)\n", sx, sy, sz, x_c, y_c, z_c);

						// Sample grid based on method by Driscoll and Healy (1994)
						double grid[NLAT][NLON];
						sample_dh1(
							RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
							V,
							x_c, y_c, z_c,
							grid,
							dx, dy, dz
						);

						// Compute the spherical harmonics expansion of the potential energy contribution
						expand_spherical_harmonics(
							grid,
							VLM_SLICE(electrode.Vlm, sx, sy, sz)
						);
					}

			free(V);
		}
	}
}

void solver_relaxation
(
	struct Electrode *electrode,
	double (*V)[RELAXATION_NX][RELAXATION_NY][RELAXATION_NZ],
	int dx,
	int dy,
	int dz,
	int max_iterations,
	int chebyshev_acceleration
)
{
	printf("Constructing relaxation problem...\n");

	for (int x = 0; x < RELAXATION_NX; x++)
		for (int y = 0; y < RELAXATION_NY; y++)
			for (int z = 0; z < RELAXATION_NZ; z++)
				if (
					point_in_polygon_zslice(x - RELAXATION_NX/2, y - RELAXATION_NY/2, z, electrode->n_vertices, electrode->vertices, 0)
				)
					(*V)[x][y][z] = 1.0;

	// Poisson equation-tuned constant vs. Chebyshev-accelerated estimate for omega
	double rho = 1.0;
	double omega = chebyshev_acceleration ? 1.0 : 1.98;

	// Residual tracking
	double xi_max = 0.0;
	double xi_it_max_prev = 0.0;

	printf("Solving relaxation problem...\n");

	for (int it = 0; it < max_iterations; it++)
	{
		double xi_it_max = 1E-13;

		for (int p = 0; p < 2; p++)
		{
			for (int x = 0; x < RELAXATION_NX-1; x++)
				for (int y = 0; y < RELAXATION_NY-1; y++)
					for (int z = 0; z < RELAXATION_NZ-1; z++)
					{
						if (
							!point_in_polygon_zslice(x - RELAXATION_NX/2, y - RELAXATION_NY/2, z, electrode->n_vertices, electrode->vertices, 1) &&
							(x != 0 && y != 0 && z != 0) &&
							((int)(x + y + z) % 2 == p)
						)
						{
							double xi = (
								(*V)[x - dx][y][z] + (*V)[x + dx][y][z] +
								(*V)[x][y - dy][z] + (*V)[x][y + dy][z] +
								(*V)[x][y][z - dz] + (*V)[x][y][z + dz]
							) - 6 * (*V)[x][y][z];

							if (fabs(xi) > fabs(xi_it_max))
								xi_it_max = xi;

							(*V)[x][y][z] += 1.0/6.0 * omega * xi;
						}
					}

			if (rho > 1E-9)
			{
				rho = 1.0 - 1E-6;
				omega = (it == 0 && p == 0) ? 1 / (1 - 0.5 * rho*rho) : 1 / (1 - 0.25 * rho*rho * omega);
			}
		}

		if (fabs(xi_it_max - xi_it_max_prev) < 1E-9)
		{
			printf("Stopping relaxation loop early at iteration %d due to small residuals\n", it);
			break;
		}

		xi_max = MAX(xi_it_max, xi_max);
		xi_it_max_prev = xi_it_max;
	}
}

// @TODO - consider generalizing syntax and moving to ti_utils.c for generality
void sample_dh1
(
	const int NX,
	const int NY,
	const int NZ,
	double (*V)[NX][NY][NZ],
	double x_c,
	double y_c,
	double z_c,
	double grid[NLAT][NLON],
	double dx,
	double dy,
	double dz
)
{
	for (int i = 0; i < NLAT; i++) {
		double phi = (double)i/NLAT * PI;

		// Precompute trigonometric quantities to reduce cost
		double R_sin_phi = SPH_R * sin(phi);
		double R_cos_phi = SPH_R * cos(phi);

		for (int j = 0; j < NLON; j++) {
			double theta = (double)j/NLON * PI;

			double sample_coord[3] = { x_c + R_sin_phi * cos(theta), y_c + R_sin_phi * sin(theta), z_c + R_cos_phi };

			interpolate_3d(
				RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ,
				V,
				sample_coord,
				&grid[i][j],
				dx, dy, dz
			);

			// @TEST - print non-zero values
			// if (fabs(grid[i][j]) > 1E-6)
			// 	printf("V(%f,%f,%f)=%.13f\n", SPH_R, phi, theta, grid[i][j]);
		}
	}
}

// @TODO - generalize syntax and move to ti_utils.c for generality
void expand_spherical_harmonics
(
	double grid[NLAT][NLON],
	double *alm
)
{
	// Define new variables so that Fortran recognizes the types
	int nlat = NLAT;
	int nlon = NLON;
	int lmax = LMAX;

	// Call the Fortran subroutine
	// @TODO - get the complex subroutine working
	// compute_shcoeffs_cmplx((double*) grid, &nlat, &nlon, &lmax, alm);
	compute_shcoeffs_real((double*) grid, &nlat, &nlon, &lmax, alm);

	// @TEST - print non-zero coefficients
	for (int l = 0; l <= LMAX; l++) {
		for (int m = -l; m <= l; m++) {
			double alm_real = 0.0;
			double alm_imag = 0.0;

			if (m < 0)
			{
				// Y_l,-m = (-1)^m (Y_l,m)*
				alm_real = pow(-1, m) * alm[2 * (l * (LMAX + 1) + abs(m))];
				alm_imag = pow(-1, m) * -1 * alm[2 * (l * (LMAX + 1) + abs(m)) + 1];
			}
			else
			{
				alm_real = alm[2 * (l * (LMAX + 1) + m)];
				alm_imag = alm[2 * (l * (LMAX + 1) + m) + 1];
			}

			if (fabs(alm_real) > 1E-6 || fabs(alm_imag) > 1E-6)
				printf("a_lm[%d,+%d] = %f + %fi\n", l, m, alm_real, alm_imag);
		}
	}
}

static inline int vlm_index(
    int l,
    int m
)
{
    return 2*(l*(LMAX+1)+m);
}

static double dPlm_dx(
    int l,
    int m,
    double x
)
{
    if (l == 0)
        return 0.0;

    double Plm = gsl_sf_legendre_sphPlm(l, m, x);
    double Plm_prev = gsl_sf_legendre_sphPlm(l-1, m, x);

    return (l*x*Plm - (l+m)*Plm_prev) / (x*x-1.0);
}

static double dPlm_dtheta(
    int l,
    int m,
    double theta
)
{
	// @TODO - does this need to be made more consistent?
    const double h = 1E-6;

    double c1 = cos(theta+h);
	double c2 = cos(theta-h);

    double P1 = gsl_sf_legendre_sphPlm(l, m, c1);
    double P2 = gsl_sf_legendre_sphPlm(l, m, c2);

    return (P1-P2)/(2*h);
}

static double d2Plm_dtheta2(
    int l,
    int m,
    double theta
)
{
	// @TODO - does this need to be made more consistent?
    double h = 1E-6;

    double p1 = gsl_sf_legendre_sphPlm(l,m,cos(theta+h));

    double p0 = gsl_sf_legendre_sphPlm(l,m,cos(theta));

    double pm1 = gsl_sf_legendre_sphPlm(l,m,cos(theta-h));

    return (p1-2*p0+pm1) / (h*h);
}

void evaluate_electrode_potential(
    struct Electrode *electrode,
    double position[3],
    double *phi
)
{
	*phi=0.0;

    for(int sx = 0; sx < NSPH_X; sx++)
    for(int sy = 0; sy < NSPH_Y; sy++)
    for(int sz = 0; sz < NSPH_Z; sz++)
    {
        double cx = sx*SPH_SPACING + SPH_R;
        double cy = sy*SPH_SPACING + SPH_R;
        double cz = sz*SPH_SPACING + SPH_Z_MIN + 1;

        double dx = position[0] - cx;
		double dy = position[1] - cy;
        double dz = position[2] - cz;

        double r = sqrt(dx*dx+dy*dy+dz*dz);

        if (r > SPH_R || r == 0)
            continue;

        double theta = acos(dz/r);
        double phi_angle = atan2(dy, dx);
        double costheta = cos(theta);

        double Y[gsl_sf_legendre_array_n(LMAX)];

        gsl_sf_legendre_array_e(
            GSL_SF_LEGENDRE_SPHARM,
            LMAX,
            costheta,
            -1,
            Y
        );

        int Vlm_len = electrode->Vlm_len;
        double (*Vlm)[Vlm_len] = electrode->Vlm;

        double (*Vlm_sphere)[(LMAX+1)*(LMAX+1)*2] = VLM_SLICE(&Vlm, sx, sy, sz);

        for (int l = 0; l <= LMAX; l++)
			for (int m = 0; m <= l; m++)
            {
                int idx = vlm_index(l,m);

                double A = (*Vlm_sphere)[idx];
                double B = (*Vlm_sphere)[idx+1];

                double angular = A*cos(m*phi_angle) + B*sin(m*phi_angle);

                *phi += pow(r,l) * Y[l*(l+1)/2+m] * angular;
            }
    }
}

void evaluate_electrode_gradient(
    struct Electrode *electrode,
    double position[3],
    double grad_phi[3]
)
{
    grad_phi[0] = 0;
    grad_phi[1] = 0;
    grad_phi[2] = 0;

    for (int sx = 0; sx < NSPH_X; sx++)
    for (int sy = 0; sy < NSPH_Y; sy++)
    for (int sz = 0; sz < NSPH_Z; sz++)
    {
        double c[3]= { sx*SPH_SPACING+SPH_R, sy*SPH_SPACING+SPH_R, sz*SPH_SPACING+SPH_Z_MIN+1 };

        double x = position[0] - c[0];
        double y = position[1] - c[1];
        double z = position[2] - c[2];

        double r = sqrt(x*x+y*y+z*z);

        if (r > SPH_R || r == 0)
            continue;

        double theta = acos(z/r);
        double phi = atan2(y,x);

        double st = sin(theta);
        double ct = cos(theta);

        double er[3]= { st*cos(phi), st*sin(phi), ct };
        double et[3]= { ct*cos(phi), ct*sin(phi), -st };
        double ep[3]= { -sin(phi), cos(phi), 0 };

        double Y[gsl_sf_legendre_array_n(LMAX)];

        gsl_sf_legendre_array_e(
            GSL_SF_LEGENDRE_SPHARM,
            LMAX,
            ct,
            -1,
            Y
        );

        double (*Vlm)[electrode->Vlm_len] = electrode->Vlm;

        double (*Vs)[(LMAX+1)*(LMAX+1)*2] = VLM_SLICE(&Vlm, sx, sy, sz);

        for(int l=0;l<=LMAX;l++)
            for(int m=0;m<=l;m++)
            {
                int idx = vlm_index(l,m);

                double A = (*Vs)[idx];
                double B = (*Vs)[idx+1];

                double angular = A*cos(m*phi) + B*sin(m*phi);
                double dangular = -A*m*sin(m*phi) + B*m*cos(m*phi);

                double Plm = Y[l*(l+1)/2+m];
                double dP = dPlm_dx(l, m, ct);

                double dtheta = -st*dP;

                double R = pow(r, l);
                double dr = l*pow(r, l-1);

                double radial = dr* Plm * angular;
                double polar = R * dtheta * angular / r;

                double azimuth = R* Plm * dangular / (r * st + 1E-30);

                for (int k = 0; k < 3; k++)
                    grad_phi[k] += radial*er[k] + polar*et[k] + azimuth*ep[k];
            }
    }
}

void evaluate_electrode_hessian_finite_differences(
    struct Electrode *electrode,
    double position[3],
    double H[3][3]
)
{
	// @TODO - factor this out
    double h = 1E-5;

    for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		H[i][j] = 0;

    for (int j = 0; j < 3; j++)
    {
        double p1[3] = { position[0], position[1], position[2] };
        double p2[3] = { position[0], position[1], position[2] };

        p1[j]+=h;
        p2[j]-=h;

        double g1[3];
        double g2[3];

        evaluate_electrode_gradient(electrode, p1, g1);
        evaluate_electrode_gradient(electrode, p2, g2);

        for (int i = 0; i < 3; i++)
            H[i][j] = (g1[i]-g2[i]) / (2 * h);
    }
}

void evaluate_electrode_hessian_analytical(
    struct Electrode *electrode,
    double position[3],
    double H[3][3]
)
{
    for (int i = 0; i < 3; i++)
	for (int j = 0; j < 3; j++)
		H[i][j] = 0.0;

    for (int sx = 0; sx < NSPH_X; sx++)
    for (int sy = 0; sy < NSPH_Y; sy++)
    for (int sz = 0; sz < NSPH_Z; sz++)
    {

        double cx = sx*SPH_SPACING + SPH_R;
        double cy = sy*SPH_SPACING + SPH_R;
        double cz = sz*SPH_SPACING + SPH_Z_MIN + 1;

        double x = position[0] - cx;
        double y = position[1] - cy;
        double z = position[2] - cz;

        double r = sqrt(x*x+y*y+z*z);

        if (r > SPH_R || r == 0)
            continue;

        double theta = acos(z/r);
        double phi = atan2(y,x);

        double st = sin(theta);
        double ct = cos(theta);

        double er[3]= { st*cos(phi), st*sin(phi), ct };
        double et[3]= { ct*cos(phi), ct*sin(phi), -st };
        double ep[3]= { -sin(phi), cos(phi), 0 };

        double (*Vlm)[electrode->Vlm_len] = electrode->Vlm;
        double (*Vs)[(LMAX+1)*(LMAX+1)*2] = VLM_SLICE(&Vlm, sx, sy, sz);

        for (int l = 0; l <= LMAX; l++)
			for (int m = 0; m <= l; m++)
			{
				int idx = 2*(l*(LMAX+1)+m);

				double A = (*Vs)[idx];
				double B = (*Vs)[idx+1];

				double angular = A*cos(m*phi) + B*sin(m*phi);
				double angular_phi = -A*m*sin(m*phi) + B*m*cos(m*phi);
				double angular_phiphi = -m * m * angular;

				double P = gsl_sf_legendre_sphPlm(l, m, ct);
				double Ptheta = -st * dPlm_dx(l, m, ct);
				double Pthetatheta = d2Plm_dtheta2(l, m, theta);

				double R = pow(r, l);

				double f_rr = l*(l-1) * pow(r, l-2) * P * angular;
				double f_rtheta = l * pow(r, l-1) * Ptheta * angular;
				double f_rphi = l * pow(r, l-1) * P * angular_phi;
				double f_thetatheta = R * Pthetatheta * angular;
				double f_thetaphi = R * Ptheta * angular_phi;
				double f_phiphi = R * P * angular_phiphi;

				double basis[3][3] =
				{
					{er[0], et[0], ep[0]},
					{er[1], et[1], ep[1]},
					{er[2], et[2], ep[2]}
				};

				double Hs[3][3] =
				{
					{
						f_rr,
						f_rtheta/r,
						f_rphi/(r*st+1E-30)
					},
					{
						f_rtheta/r,
						f_thetatheta/(r*r),
						f_thetaphi/(r*r*st+1E-30)
					},
					{
						f_rphi/(r*st+1E-30),
						f_thetaphi/(r*r*st+1E-30),
						f_phiphi/(r*r*st*st+1E-30)
					}
				};

				for(int i = 0; i < 3; i++)
				for(int j = 0; j < 3; j++)
					for(int a = 0; a < 3; a++)
					for(int b = 0; b < 3; b++)
					{
						H[i][j]
						+=
						basis[i][a]
						*
						Hs[a][b]
						*
						basis[j][b];
					}
			}
    }
}


