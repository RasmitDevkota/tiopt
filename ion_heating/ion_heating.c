#include "ion_heating.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static const double HBAR = 1.054571817e-34;

static double ion_gamma_one_mode(double q, double m, double omega_sec, const double *e,
                                 const spectrum_opts_t *opt) {
    double *f_hz = NULL;
    double *psd = NULL;
    int nb = ion_compute_psd(e, opt, &f_hz, &psd);
    if (nb < 0) return -1.0;

    double S = ion_psd_at_omega(f_hz, psd, nb, omega_sec);
    free(f_hz);
    free(psd);

    if (omega_sec <= 0.0 || m <= 0.0) return 0.0;
    double pref = (q * q) / (4.0 * m * HBAR * omega_sec);
    return pref * S;
}

void ion_spectrum_opts_default(spectrum_opts_t *o) {
    o->dt = 1e-7;
    o->n = 4096;
    o->use_hann = 1;
    o->subtract_mean = 1;
}

static void fft_radix2(double *x_re, double *x_im, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = x_re[i], ti = x_im[i];
            x_re[i] = x_re[j];
            x_im[i] = x_im[j];
            x_re[j] = tr;
            x_im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        double wlen_re = cos(ang);
        double wlen_im = sin(ang);
        for (int i = 0; i < n; i += len) {
            double w_re = 1.0, w_im = 0.0;
            for (int k = 0; k < len / 2; k++) {
                int i0 = i + k;
                int i1 = i0 + len / 2;
                double u_re = x_re[i0], u_im = x_im[i0];
                double v_re = x_re[i1] * w_re - x_im[i1] * w_im;
                double v_im = x_re[i1] * w_im + x_im[i1] * w_re;
                x_re[i0] = u_re + v_re;
                x_im[i0] = u_im + v_im;
                x_re[i1] = u_re - v_re;
                x_im[i1] = u_im - v_im;
                double nw_re = w_re * wlen_re - w_im * wlen_im;
                double nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

static double hann(int i, int n) {
    return 0.5 * (1.0 - cos(2.0 * M_PI * i / (n - 1)));
}

int ion_compute_psd(const double *e, const spectrum_opts_t *opt,
                    double **f_hz, double **psd) {
    int n = opt->n;
    if (n < 4 || (n & (n - 1)) != 0) return -1;

    double *re = (double *)calloc((size_t)n * 2, sizeof(double));
    double *im = re + n;
    if (!re) return -1;

    double mean = 0.0;
    if (opt->subtract_mean) {
        for (int i = 0; i < n; i++) mean += e[i];
        mean /= n;
    }

    double wsum2 = 0.0;
    for (int i = 0; i < n; i++) {
        double w = opt->use_hann ? hann(i, n) : 1.0;
        re[i] = (e[i] - mean) * w;
        im[i] = 0.0;
        wsum2 += w * w;
    }

    fft_radix2(re, im, n);

    double Fs = 1.0 / opt->dt;
    int nb = n / 2 + 1;
    double *fh = (double *)malloc((size_t)nb * sizeof(double));
    double *ps = (double *)malloc((size_t)nb * sizeof(double));
    if (!fh || !ps) {
        free(fh);
        free(ps);
        free(re);
        return -1;
    }

    double denom = Fs * (double)n * wsum2;
    if (denom <= 0.0) denom = 1.0;

    for (int k = 0; k < nb; k++) {
        fh[k] = k * Fs / (double)n;
        double p2 = re[k] * re[k] + im[k] * im[k];
        if (k == 0 || k == n / 2)
            ps[k] = p2 / denom;
        else
            ps[k] = 2.0 * p2 / denom;
    }

    free(re);
    *f_hz = fh;
    *psd = ps;
    return nb;
}

double ion_psd_at_omega(const double *f_hz, const double *psd, int n_bins,
                        double omega_sec) {
    double f_target = omega_sec / (2.0 * M_PI);
    if (n_bins < 2) return 0.0;
    if (f_target <= f_hz[0]) return psd[0];
    if (f_target >= f_hz[n_bins - 1]) return psd[n_bins - 1];
    for (int k = 0; k < n_bins - 1; k++) {
        if (f_hz[k] <= f_target && f_target <= f_hz[k + 1]) {
            double t = (f_target - f_hz[k]) / (f_hz[k + 1] - f_hz[k]);
            return psd[k] * (1.0 - t) + psd[k + 1] * t;
        }
    }
    return 0.0;
}

static const double ION_LAB_AXES[3][3] = {
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

void ion_field_from_potential3(ion_potential_fn3 V, void *v_ctx, double t,
                               const double r[3], double h, double e_out[3]) {
    if (h <= 0.0) h = 1e-9;
    double rp[3], rm[3];
    for (int j = 0; j < 3; j++) {
        rp[0] = r[0];
        rp[1] = r[1];
        rp[2] = r[2];
        rm[0] = r[0];
        rm[1] = r[1];
        rm[2] = r[2];
        rp[j] += h;
        rm[j] -= h;
        e_out[j] = -(V(t, rp, v_ctx) - V(t, rm, v_ctx)) / (2.0 * h);
    }
}

void ion_sample_modes_along_path_3d(double t0, const spectrum_opts_t *opt,
                                    ion_path_fn3 path, void *path_ctx,
                                    ion_field_fn3 field, void *field_ctx,
                                    const ion_params3_t *ion3,
                                    double *e0, double *e1, double *e2) {
    const double (*ax)[3] = ion3->axes ? ion3->axes : ION_LAB_AXES;
    double r[3], E[3];

    for (int i = 0; i < opt->n; i++) {
        double t = t0 + i * opt->dt;
        path(t, path_ctx, r);
        field(t, r, field_ctx, E);
        e0[i] = ax[0][0] * E[0] + ax[0][1] * E[1] + ax[0][2] * E[2];
        e1[i] = ax[1][0] * E[0] + ax[1][1] * E[1] + ax[1][2] * E[2];
        e2[i] = ax[2][0] * E[0] + ax[2][1] * E[1] + ax[2][2] * E[2];
    }
}

void ion_heating_rates_gamma3(const ion_params3_t *ion3,
                              const double *e0, const double *e1, const double *e2,
                              const spectrum_opts_t *opt, double gamma_out[3]) {
    const double *ek[3] = {e0, e1, e2};
    for (int k = 0; k < 3; k++) {
        double g = ion_gamma_one_mode(ion3->q, ion3->m, ion3->omega_sec[k], ek[k], opt);
        gamma_out[k] = (g < 0.0) ? 0.0 : g;
    }
}

void ion_heating_rates_from_path_field_3d(double t0, const ion_params3_t *ion3,
                                          const spectrum_opts_t *opt,
                                          ion_path_fn3 path, void *path_ctx,
                                          ion_field_fn3 field, void *field_ctx,
                                          double gamma_out[3]) {
    double *b0 = (double *)malloc((size_t)opt->n * sizeof(double) * 3);
    if (!b0) {
        gamma_out[0] = gamma_out[1] = gamma_out[2] = -1.0;
        return;
    }
    double *b1 = b0 + opt->n;
    double *b2 = b1 + opt->n;

    ion_sample_modes_along_path_3d(t0, opt, path, path_ctx, field, field_ctx, ion3,
                                   b0, b1, b2);
    ion_heating_rates_gamma3(ion3, b0, b1, b2, opt, gamma_out);
    free(b0);
}

double ion_heating_total_gamma3(const double gamma[3]) {
    return gamma[0] + gamma[1] + gamma[2];
}
