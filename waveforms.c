#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <nlopt.h>

#include "waveforms.h"
#include "data_structures.h"
#include "ti_utils.h"

double waveform_cost_function(
    unsigned n,
    const double *x,
    double *grad,
    void *transport_waveform_obj
)
{
    struct TransportWaveform *w =
        (struct TransportWaveform *)transport_waveform_obj;


    double cost = 0.0;


    /*
     * Penalize large voltages
     */
    for(int i=0;i<n;i++)
    {
        cost += x[i]*x[i];
    }


    /*
     * Penalize waveform roughness:
     *
     * sum(V(t+1)-2V(t)+V(t-1))^2
     */
    for(int t=1;t<w->n_waveform_steps-1;t++)
    {
        for(int e=0;e<w->n_electrodes;e++)
        {
            int k =
              t*w->n_electrodes+e;

            double d2 =
                x[k+w->n_electrodes]
              -2*x[k]
              +x[k-w->n_electrodes];


            cost += 1000*d2*d2;
        }
    }


    if(grad)
    {
        for(int i=0;i<n;i++)
            grad[i]=0;
    }


    return cost;
}

double transport_waveform_constraint(
    unsigned n,
    const double *x,
    double *grad,
    void *data
)
{
    transport_waveform_constraint_data *d = data;

    double max_voltage_individual = d->max_voltage_individual;
    double max_voltage_total = d->max_voltage_total;

	double constraint_value = 0.0;

	constraint_value -= fabs(max_voltage_total);

	for (int s = 0; s <= d->n_steps; s++)
		for (int e = 0; e <= d->n_electrodes; e++)
		{
			double V = x[s*d->n_electrodes + e];

			constraint_value += fabs(V);

			if (grad)
				grad[s*d->n_electrodes + e] = -V;
		}


    return constraint_value;
}

void waveform_potential_derivatives
(
    struct Trap *trap,
    double *V,
    double position[3],
    double *grad_phi,
    double H[3][3]
)
{
    for(int k=0;k<3;k++)
        grad_phi[k]=0;

    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            H[i][j]=0;

    for(int e=0;e<trap->n_electrodes;e++)
    {
        double electrode_gradient[3];

        double electrode_H[3][3];


        evaluate_electrode_derivatives(
            &trap->electrodes[e],
            position,
            electrode_gradient,
            electrode_H
        );


        double Ve =
            V[e];


        for(int k=0;k<3;k++)
            grad_phi[k]
                += Ve*electrode_gradient[k];


        for(int i=0;i<3;i++)
            for(int j=0;j<3;j++)
                H[i][j]
                    += Ve*electrode_H[i][j];

    }
}

double constraint_field_i
(
    unsigned n,
    const double *x,
    double *grad,
    void *data
)
{
    struct transport_waveform_constraint_data *d = data;

	int i = d->i;

    double *V = (double *) &x[d->voltage_offset];

    double g[3];
    double H[3][3];

    waveform_potential_derivatives(
        d->trap,
        V,
        d->position,
        g,
        H
    );

    if (grad)
    {
        /*
         * TODO:
         * populate derivative wrt every voltage
         */
        memset(
            grad,
            0,
            n*sizeof(double)
        );
    }


    return g[i];
}

double constraint_hessian_ij
(
    unsigned n,
    const double *x,
    double *grad,
    void *data
)
{
    struct transport_waveform_constraint_data *d = data;

	int i = d->i;
	int j = d->j;

    double g[3];
    double H[3][3];

    waveform_potential_derivatives(
        d->trap,
        (double *) &x[d->voltage_offset],
        d->position,
        g,
        H
    );

    if (grad)
        memset(grad, 0, n*sizeof(double));

    return H[0][2];
}

double constraint_axial_frequency
(
    unsigned n,
    const double *x,
    double *grad,
    void *data
)
{
    struct transport_waveform_constraint_data *d = data;


    double g[3];
    double H[3][3];


    waveform_potential_derivatives(
        d->trap,
        (double *)&x[d->voltage_offset],
        d->position,
        g,
        H
    );


    double target =
        d->mass *
        d->omega[2] *
        d->omega[2]
        /
        d->charge;



    if(grad)
        memset(
            grad,
            0,
            n*sizeof(double)
        );


    return H[2][2]-target;
}

void optimize_primitive
(
	int max_iterations
)
{
	// @TODO - construct constraints
	// @TODO - construct initial potential vector
	
    double min_cost = 99999;
    const int dim = 12;

    printf("- Constructing optimization problem...\n");

    // nlopt setup
    nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);

    printf("Setting constraints...\n");
    // @TODO - constraints

	for(int t=0;t<waveform->n_waveform_steps;t++)
	{
		struct transport_waveform_constraint_data *cx = malloc(sizeof(*cx));

		cx->trap=trap;

		cx->timestep=t;

		cx->voltage_offset =
			t*trap->n_electrodes;


		memcpy(
			cx->position,
			trajectory[t],
			3*sizeof(double)
		);


		nlopt_add_equality_constraint(
			opt,
			constraint_field_x,
			cx,
			1e-8
		);


		nlopt_add_equality_constraint(
			opt,
			constraint_field_y,
			cx,
			1e-8
		);


		nlopt_add_equality_constraint(
			opt,
			constraint_field_z,
			cx,
			1e-8
		);


		nlopt_add_equality_constraint(
			opt,
			constraint_hessian_xz,
			cx,
			1e-8
		);


		nlopt_add_equality_constraint(
			opt,
			constraint_hessian_yz,
			cx,
			1e-8
		);


		nlopt_add_equality_constraint(
			opt,
			constraint_axial_frequency,
			cx,
			1e-8
		);
	}

    printf("Setting bounds...\n");
    // @TODO - bounds

    printf("Setting stopping criteria...\n");
    // @TODO - tolerances

	// @TEST - limit number of evaluations to 1 for testing purposes
	nlopt_set_maxeval(opt, 1);

    printf("Constructing initial guess...\n");
    // Generate initial guess vector
    struct TransportWaveform transport_waveform;
	// @TODO - construct initial transport waveform

    double x[dim];
    for (int i = 0; i < dim; i++)
    {
        x[i] = 0.0;
    }

    printf("Setting objectives...\n");
    nlopt_set_min_objective(opt, waveform_cost_function, &transport_waveform);

    printf("Calling objective once...\n");
    waveform_cost_function(
      dim,
      x,
      NULL,
      &transport_waveform
    );

    printf("Optimizing...\n");
    if (nlopt_optimize(opt, x, &min_cost) < 0)
    {
        printf("nlopt failed!\n");
    }
    else
    {
        printf("Found minimum at f(");
        for (int k = 0; k < dim; k++)
        {
            printf("%g", x[k]);

            if (k < dim-1)
            {
                printf(",");
            }
        }
        printf(") = %0.10g\n", min_cost);
    }
}

void solve_waveform
(
)
{
}

// @TODO - restructure as a void func
struct TransportWaveform* transport_program_to_waveform
(
	struct Trap *trap,
	struct TransportProgram *transport_program,
	int *n_waveform_steps_list
)
{
	// Prepare waveform object
	struct TransportWaveform *waveform = malloc(sizeof(struct TransportWaveform));

	waveform->n_waveform_steps = 0;
	for (int t = 0; t < transport_program->n_transports; t++)
		waveform->n_waveform_steps += n_waveform_steps_list[t];

	waveform->voltages = calloc(waveform->n_waveform_steps, sizeof(double));

	// Set up constraints
	double *V_F = malloc(trap->n_electrodes * sizeof(double));
	double *Psi_F = malloc(trap->n_electrodes * sizeof(double));

	double *P = malloc(trap->n_electrodes * sizeof(double));

	double *C_1 = malloc(3 * sizeof(double));
	double *C_2 = malloc(3 * sizeof(double));
	double *D = malloc(3 * sizeof(double));
	for (int i = 0; i <= 3; i++)
	{

	}

	solve_waveform(
		//double (*V)[]
	);

	return waveform;
}

void interpolate_waveform
(
	struct Trap *trap,
	struct TransportWaveform *waveform_coarse,
	struct TransportWaveform *waveform_fine,
	double dx
)
{
    for (int s = 0; s < waveform_fine->n_waveform_steps; s++)
    {
        interpolate_1d(
            waveform_coarse>n_waveform_steps * trap->n_electrodes,
            waveform_coarse>voltages,
            s,
            waveform_fine->voltages[s],
            dx
        );
    }
}

void compose_waveforms
(
	int n_waveforms,
	struct TransportWaveform *waveforms,
	double *waveform_scales // @TODO - implement support for waveform_scales not equal to unity
)
{
	struct TransportWaveform *composite_waveform = malloc(sizeof(struct TransportWaveform));

	composite_waveform->n_waveform_steps = 0;
	for (int w = 0; w < n_waveforms; w++)
		composite_waveform->n_waveform_steps += waveforms[w].n_waveform_steps;

	// @TODO - make sure that this memory is freed appropriately
	composite_waveform->voltages = calloc(composite_waveform->n_waveform_steps, sizeof(double));

	int step = 0;
	for (int w = 0; w < n_waveforms; w++)
		for (int substep = 0; substep < waveforms[w].n_waveform_steps; substep++)
			(*composite_waveform->voltages)[step] += (*waveforms[w].voltages)[substep];

	return;
}

// @TODO - update
void compute_secular_frequencies
(
	double q,
	double m,
	double V_DC,
	double V_RF,
	double Omega,
	double Z_0,
	double *omega
)
{
	double a_z = -4 * q * V_DC / (m * pow(ION_HEIGHT, 2) * pow(Omega, 2));
	double a_x = -0.5 * a_z;
	double a_y = -0.5 * a_z;

	double q_x = 2 * q * V_RF / (m * pow(Z_0, 2) * pow(Omega, 2));
	double q_y = -q_x;
	double q_z = 0.0;

	omega[0] = Omega/2 * sqrt(a_x + 0.5 * pow(q_x, 2));
	omega[1] = Omega/2 * sqrt(a_y + 0.5 * pow(q_y, 2));
	omega[2] = Omega/2 * sqrt(a_z + 0.5 * pow(q_z, 2));
}

void compute_U_pond
(
	double m,
	double omega_rf,
	double V_rf,
	double *U_rf,
	double *U_pond,
	double dx
)
{
	double (*grad_U_rf)[3] = malloc(3 * sizeof(double));

	grad_1d(
		3,
		U_rf,
		grad_U_rf,
		dx
	);

	double grad_U_rf_sq = 0.0;
	dot(
		3,
		grad_U_rf,
		grad_U_rf,
		&grad_U_rf_sq
	);

	// @TODO - fill in calculation
	*U_pond = 0.0;
}

