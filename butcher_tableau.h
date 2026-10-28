#ifndef TIOPT_BUTCHER_TABLEAU_H
#define TIOPT_BUTCHER_TABLEAU_H

#include <complex.h>
#include <stddef.h>

struct Trap;

#define BT_MATRIX_SIZE 5
/* Matrices use column-major storage: row + BT_MATRIX_SIZE * column. */
typedef struct {
    double omega_422;
    double omega_674;
    double omega_1092;

    double phi_422;
    double phi_1092;
    double phi_674;

    double delta_422;
    double delta_1092;
    double delta_674;
    double delta_3;

    double k_422;
    double k_1092;
    double k_674;

    double ion_v;

    double gamma_422;
    double gamma_1092;
    double gamma_674;

    /* Borrowed trap with solved Vlm coefficients and n_electrodes voltages.
     * NULL trap selects free space; NULL voltages selects zero applied voltage.
     * q is the ion charge in coulombs for SI inputs. */
    struct Trap *trap;
    const double *electrode_voltages;
    double q;
} bt_sys_param;

typedef struct {
    double t;
    double x;
    double y;
    double z;
    double v;
    double m;
} bt_dynamic_states;

typedef struct {
    double complex H[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    double complex rho[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    double complex L1[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    double complex L2[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
    double complex L3[BT_MATRIX_SIZE * BT_MATRIX_SIZE];
} bt_quantum_matrices;


/* Initialize the dimensionless decay example and its mixed density matrix. */
void bt_sys_param_init(bt_sys_param *param);
void bt_state_init(bt_dynamic_states *state);
void bt_matrices_init(const bt_sys_param *param, bt_quantum_matrices *matrices);

/* Explicit Runge–Kutta tableau. Arrays are borrowed; a is row-major
 * stages-by-stages (unlike the column-major quantum matrices).
 * a[i*stages+j] must be zero for j >= i. Array lengths are caller-owned.
 * Stage count does not determine order; supply coefficients for your method. */
typedef struct {
    size_t stages;
    const double *a;
    const double *b;
    const double *c;
} bt_tableau;

extern const bt_tableau BT_RK4;

/* Arbitrary-stage explicit RK step. Returns 0 on success, -1 for invalid
 * arguments/nonfinite coefficients or an implicit tableau, -2 on allocation
 * failure. Errors leave state and matrices unchanged. Allocates stage storage.
 * Does not check order conditions or choose a stable step size. */
int bt_rk_step(bt_dynamic_states *state, bt_sys_param *param,
               bt_quantum_matrices *matrices, double step,
               const bt_tableau *tableau);

/* Allocation-free variant for repeated steps. Reuse caller-owned storage of
 * at least tableau->stages * BT_MATRIX_SIZE * BT_MATRIX_SIZE complex entries.
 * Workspace must not overlap state, params, matrices, or coefficient arrays.
 * Returns 0 or -1; workspace contents are scratch and need no initialization. */
int bt_rk_step_workspace(bt_dynamic_states *state, bt_sys_param *param,
                         bt_quantum_matrices *matrices, double step,
                         const bt_tableau *tableau, double complex *workspace,
                         size_t entries);

/* One classical RK4 Butcher-tableau step of the Lindblad equation.
 * Advances state->t and matrices->rho; x, y, z, v and m are held fixed.
 * step is in the same time units as the supplied rates (seconds for SI rates).
 * No allocation; all arguments must be non-null. */
void bt_rk4_step(bt_dynamic_states *state, bt_sys_param *param,
                 bt_quantum_matrices *matrices, double step);
int bt_decay_demo(void);

#endif
