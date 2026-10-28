#include "butcher_tableau.h"
#include "electrodynamics.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

static const double BT_HBAR = 1.054571817e-34;

static int get_index(int row, int col) { return row + BT_MATRIX_SIZE * col; }

/* Accumulate scale * A * B without allocating a product or dagger matrix.
 * out must not overlap either input. */
static void product_add(const double complex *a, const double complex *b,
                        int dagger_a, int dagger_b, double complex scale,
                        double complex *out)
{
    for (int j = 0; j < BT_MATRIX_SIZE; ++j)
        for (int i = 0; i < BT_MATRIX_SIZE; ++i) {
            double complex sum = 0;
            for (int k = 0; k < BT_MATRIX_SIZE; ++k) {
                double complex av = dagger_a ? conj(a[get_index(k,i)]) : a[get_index(i,k)];
                double complex bv = dagger_b ? conj(b[get_index(j,k)]) : b[get_index(k,j)];
                sum += av * bv;
            }
            out[get_index(i,j)] += scale * sum;
        }
}

void bt_matrices_init(const bt_sys_param *p, bt_quantum_matrices *qm)
{
    memset(qm, 0, sizeof(*qm));
    qm->L1[get_index(0,1)] = sqrt(p->gamma_422);
    qm->L2[get_index(2,1)] = sqrt(p->gamma_1092);
    qm->L3[get_index(0,4)] = sqrt(p->gamma_674);
    qm->rho[get_index(1,1)] = 0.75;
    qm->rho[get_index(4,4)] = 0.25;
}

/* Decay-only test: dimensionless rates, no laser driving or detuning. */
void bt_sys_param_init(bt_sys_param *param_in) {
    *param_in = (bt_sys_param){0};
    param_in->gamma_422 = 1.0;
    param_in->gamma_1092 = 2.0;
    param_in->gamma_674 = 0.5;
}

//initialize states here 
void bt_state_init(bt_dynamic_states *states_in) {
    states_in -> t = 0.0;
    states_in -> x = 0.0;
    states_in -> y = 0.0;
    states_in -> z = 0.0;
    states_in -> v = 0.0; 
    states_in -> m = 1.0;
}

static double get_K(
    double m,
    double v
) {
    return m * v * v / 2;
}

static void construct_hamiltonian(
    bt_dynamic_states *s, //states
    bt_sys_param *p, //parameters
    double complex *H //Hamiltonian
) {

    for(int i = 0; i < BT_MATRIX_SIZE * BT_MATRIX_SIZE; i++) {
        H[i] = 0.0 + 0.0*I;
    }

    double complex H01 = p -> omega_422 * cexp(-I * p -> phi_422);
    double complex H04 = p -> omega_674 * cexp(-I * p -> phi_674);
    double complex H11 = 2 * (p -> delta_422 - p -> k_422 * s -> v);
    double complex H12 = p -> omega_1092 * cexp(-I * p -> phi_1092);
    double complex H22 = 2 * (p -> delta_1092 - p -> k_1092 * s -> v);
    double complex H33 = 2 * p -> delta_3;
    double complex H44 = 2 * (p -> delta_674 - p -> k_674 * s -> v);

    //off diagonal
    H[get_index(0,1)] = H01;
    H[get_index(1,0)] = conj(H01);

    H[get_index(0,4)] = H04;
    H[get_index(4,0)] = conj(H04);

    H[get_index(1,2)] = H12;
    H[get_index(2,1)] = conj(H12);

    H[get_index(1,1)] = H11;
    H[get_index(2,2)] = H22;
    H[get_index(3,3)] = H33;
    H[get_index(4,4)] = H44;

    for (int i = 0; i < BT_MATRIX_SIZE * BT_MATRIX_SIZE; i++) {
        H[i] = H[i] * BT_HBAR / 2.0;
    }

    //add kinetic energy and potential energy
    double potential = 0.0;
    if (p->trap && p->electrode_voltages && p->q != 0.0) {
        double position[3] = {s->x, s->y, s->z};
        for (int e = 0; e < p->trap->n_electrodes; ++e) {
            if (p->electrode_voltages[e] == 0.0)
                continue;
            double phi;
            evaluate_electrode_potential(&(*p->trap->electrodes)[e],
                                         position, &phi);
            potential += p->electrode_voltages[e] * phi;
        }
    }
    double energy = get_K(s->m, s->v) + p->q * potential;

    for (int i = 0; i < BT_MATRIX_SIZE; ++i) { 
        H[get_index(i, i)] += energy;
    }   
    
}

static void eval_F(const double complex *h, const double complex *rho,
                   const bt_quantum_matrices *qm, double complex *increment)
{
    memset(increment, 0, sizeof(qm->rho));
    /* Form the commutator before scaling, preserving cancellation of common
     * diagonal energy offsets. */
    product_add(h, rho, 0, 0, 1, increment);
    product_add(rho, h, 0, 0, -1, increment);
    for (int n = 0; n < BT_MATRIX_SIZE * BT_MATRIX_SIZE; ++n)
        increment[n] *= -I / BT_HBAR;

    double complex temp[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    const double complex *collapse[] = {qm->L1, qm->L2, qm->L3};
    for (size_t l = 0; l < 3; ++l) {
        const double complex *op = collapse[l];
        memset(temp, 0, sizeof(temp));
        product_add(op, rho, 0, 0, 1, temp);
        product_add(temp, op, 0, 1, 1, increment);
        memset(temp, 0, sizeof(temp));
        product_add(op, op, 1, 0, 1, temp);
        product_add(temp, rho, 0, 0, -0.5, increment);
        product_add(rho, temp, 0, 0, -0.5, increment);
    }
}

static const double rk4_a[16] = {
    0, 0, 0, 0, 0.5, 0, 0, 0,
    0, 0.5, 0, 0, 0, 0, 1, 0
};
static const double rk4_b[4] = {1.0/6, 1.0/3, 1.0/3, 1.0/6};
static const double rk4_c[4] = {0, 0.5, 0.5, 1};
const bt_tableau BT_RK4 = {4, rk4_a, rk4_b, rk4_c};

/* Shared stage engine; the RK4 wrapper supplies stack storage. */
static void rk_step_with_storage(bt_dynamic_states *s, bt_sys_param *p,
                                 bt_quantum_matrices *qm, double step,
                                 const bt_tableau *tableau,
                                 double complex (*k)[BT_MATRIX_SIZE * BT_MATRIX_SIZE])
{
    const size_t stages = tableau->stages;
    bt_dynamic_states stage = *s;
    double complex trial[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    double complex h[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    for (size_t i = 0; i < stages; ++i) {
        for (int n = 0; n < BT_MATRIX_SIZE * BT_MATRIX_SIZE; ++n) {
            trial[n] = qm->rho[n];
            for (size_t j = 0; j < i; ++j)
                trial[n] += step * tableau->a[i*stages+j] * k[j][n];
        }
        stage.t = s->t + tableau->c[i] * step;
        construct_hamiltonian(&stage, p, h);
        eval_F(h, trial, qm, k[i]);
    }
    for (int n = 0; n < BT_MATRIX_SIZE * BT_MATRIX_SIZE; ++n)
        for (size_t i = 0; i < stages; ++i)
            qm->rho[n] += step * tableau->b[i] * k[i][n];
    s->t += step;
    construct_hamiltonian(s, p, qm->H);
}

static int validate_step(bt_dynamic_states *s, bt_sys_param *p,
                          bt_quantum_matrices *qm, double step, const bt_tableau *tableau)
{
    if (!s || !p || !qm || !tableau || !isfinite(step) ||
        !tableau->stages || !tableau->a || !tableau->b || !tableau->c)
        return -1;
    size_t stages = tableau->stages;
    /* Guard both flattened tableau indexing and stage-buffer allocation. */
    if (stages > SIZE_MAX / stages ||
        stages > SIZE_MAX / (BT_MATRIX_SIZE * BT_MATRIX_SIZE * sizeof(double complex)))
        return -1;
    for (size_t i = 0; i < stages; ++i) {
        if (!isfinite(tableau->b[i]) || !isfinite(tableau->c[i]))
            return -1;
        for (size_t j = 0; j < stages; ++j) {
            double a = tableau->a[i*stages+j];
            if (!isfinite(a) || (j >= i && a != 0))
                return -1;
        }
    }
    return 0;
}

int bt_rk_step_workspace(bt_dynamic_states *s, bt_sys_param *p,
                         bt_quantum_matrices *qm, double step,
                         const bt_tableau *tableau, double complex *workspace,
                         size_t entries)
{
    if (validate_step(s, p, qm, step, tableau) != 0) return -1;
    if (!workspace || entries < tableau->stages * BT_MATRIX_SIZE * BT_MATRIX_SIZE)
        return -1;
    rk_step_with_storage(s, p, qm, step, tableau,
                        (double complex (*)[BT_MATRIX_SIZE * BT_MATRIX_SIZE])workspace);
    return 0;
}

int bt_rk_step(bt_dynamic_states *s, bt_sys_param *p,
               bt_quantum_matrices *qm, double step, const bt_tableau *tableau)
{
    if (validate_step(s, p, qm, step, tableau) != 0) return -1;
    double complex (*k)[BT_MATRIX_SIZE * BT_MATRIX_SIZE] =
        malloc(tableau->stages * sizeof(*k));
    if (!k) return -2;
    rk_step_with_storage(s, p, qm, step, tableau, k);
    free(k);
    return 0;
}

void bt_rk4_step(bt_dynamic_states *s, bt_sys_param *p,
                 bt_quantum_matrices *qm, double step)
{
    double complex k[4][BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    rk_step_with_storage(s, p, qm, step, &BT_RK4, k);
}

int bt_decay_demo(void)
{
    bt_sys_param p;
    bt_dynamic_states s;
    bt_quantum_matrices qm;
    bt_sys_param_init(&p);
    bt_state_init(&s);
    bt_matrices_init(&p, &qm);
    for (int i = 0; i < 1000; ++i)
        bt_rk4_step(&s, &p, &qm, 0.001);
    printf("Final time: %.6f\n", s.t);
    double complex trace = 0;
    for (int i = 0; i < BT_MATRIX_SIZE; ++i) {
        double complex pop = qm.rho[get_index(i,i)];
        printf("Population[%d]: %.10f\n", i, creal(pop));
        trace += pop;
    }
    printf("Trace: %.10f %+.10fi\n", creal(trace), cimag(trace));
    return 0;
}
