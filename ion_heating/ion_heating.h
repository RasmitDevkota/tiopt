/*
 * 3D motional heating from E(r,t) along r(t) (Turchette et al., PRA 61, 063418).
 *
 * ---------------------------------------------------------------------------
 * HOW TO USE (from your own main.c)
 * ---------------------------------------------------------------------------
 * 1) #include "ion_heating.h" and link with ion_heating.c (-lm).
 *
 * 2) Fill ion_params3_t:
 *      q, m  (C, kg)
 *      omega_sec[3]  secular angular frequencies (rad/s) for modes 0,1,2
 *      axes  NULL = lab x,y,z; or double[3][3] rows = unit vectors u_k for E·u_k
 *
 * 3) Fill spectrum_opts_t (or ion_spectrum_opts_default then edit):
 *      dt, n = sample interval and count (n must be a power of 2, e.g. 8192)
 *      use_hann, subtract_mean  (defaults: 1,1 — DC removed before spectrum)
 *
 * 4) Implement callbacks:
 *      void path(double t, void *ctx, double r[3]);
 *      void field(double t, const double r[3], void *ctx, double e[3]);  // V/m
 *    Optional: only have V(t,r)? Use ion_field_from_potential3 inside field().
 *
 * 5) Call:
 *      double gamma[3];
 *      ion_heating_rates_from_path_field_3d(t0, &ion3, &opt, path, ctx_p,
 *                                           field, ctx_f, gamma);
 *    gamma[k] is Gamma_{0->1} in 1/s for mode k. Sum: ion_heating_total_gamma3(gamma).
 *
 * 6) Run from a terminal (folder containing the .c files):
 *      gcc -O2 -std=c11 -o ion_heating_demo ion_heating.c ion_heating_demo.c -lm
 *      ./ion_heating_demo
 *    (On Windows: ion_heating_demo.exe)
 * ---------------------------------------------------------------------------
 */

#ifndef ION_HEATING_H
#define ION_HEATING_H

#include <stddef.h>

typedef struct {
    double dt;          /* sample interval (s) */
    int n;              /* number of samples (power of 2, >= 4) */
    int use_hann;       /* 1: Hann window to reduce spectral leakage */
    int subtract_mean;  /* 1: remove DC from e(t) before spectrum */
} spectrum_opts_t;

void ion_spectrum_opts_default(spectrum_opts_t *o);

/*
 * One-sided PSD of scalar e(t) in (V/m)^2/Hz. Allocates *f_hz and *psd (caller frees).
 * Returns number of bins (n/2+1), or -1 on error.
 */
int ion_compute_psd(const double *e, const spectrum_opts_t *opt,
                    double **f_hz, double **psd);

double ion_psd_at_omega(const double *f_hz, const double *psd, int n_bins,
                        double omega_sec);

typedef void (*ion_path_fn3)(double t, void *ctx, double r_out[3]);
typedef void (*ion_field_fn3)(double t, const double r[3], void *ctx, double e_out[3]);
typedef double (*ion_potential_fn3)(double t, const double r[3], void *ctx);

/* E = -grad V (central differences, step h in m). */
void ion_field_from_potential3(ion_potential_fn3 V, void *v_ctx, double t,
                               const double r[3], double h, double e_out[3]);

typedef struct {
    double q;              /* C */
    double m;              /* kg */
    double omega_sec[3];   /* rad/s */
    const double (*axes)[3]; /* NULL -> lab x,y,z; else row k = u_k */
} ion_params3_t;

void ion_sample_modes_along_path_3d(double t0, const spectrum_opts_t *opt,
                                    ion_path_fn3 path, void *path_ctx,
                                    ion_field_fn3 field, void *field_ctx,
                                    const ion_params3_t *ion3,
                                    double *e0, double *e1, double *e2);

void ion_heating_rates_gamma3(const ion_params3_t *ion3,
                              const double *e0, const double *e1, const double *e2,
                              const spectrum_opts_t *opt, double gamma_out[3]);

void ion_heating_rates_from_path_field_3d(double t0, const ion_params3_t *ion3,
                                          const spectrum_opts_t *opt,
                                          ion_path_fn3 path, void *path_ctx,
                                          ion_field_fn3 field, void *field_ctx,
                                          double gamma_out[3]);

double ion_heating_total_gamma3(const double gamma[3]);

#endif
