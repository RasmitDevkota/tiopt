#include "data_structures.h"
void grad
(
	const int f_len,
	double (*f)[f_len],
	double (*grad_f)[f_len],
	const double dx
);

void interpolate_1d
(
	const int f_len,
	double (*f)[f_len],
	double x_rel,
	double *f_x_rel,
	const double dx
);

void interpolate_3d
(
	const int f_len_x,
	const int f_len_y,
	const int f_len_z,
	double (*f)[f_len_x][f_len_y][f_len_z],
	double p_rel[3],
	double *f_p_rel,
	double dx,
	double dy,
	double dz
);

int point_in_polygon_zslice
(
	int x,
	int y,
	int z,
	int n_vertices,
	double (*vertices)[3],
	int check_bounding_box
);

void compute_hessian_3d(
	const int f_len_x,
	const int f_len_y,
	const int f_len_z,
	double (*f)[f_len_x][f_len_y][f_len_z],
	double p_rel[3],
	double (*H)[3][3],
	double h[3],
	double dx,
	double dy,
	double dz
);

int* find_step_times
(
	struct PulseProgram
);

