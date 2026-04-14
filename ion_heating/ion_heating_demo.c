/*
 * Example program (3D only). Build & run:
 *   gcc -O2 -std=c11 -o ion_heating_demo ion_heating.c ion_heating_demo.c -lm
 *   ./ion_heating_demo
 */

#include "ion_heating.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

static void path_origin(double t, void *ctx, double r[3]) {
    (void)t;
    (void)ctx;
    r[0] = r[1] = r[2] = 0.0;
}

typedef struct {
    double Eamp;
    double wx, wy, wz;
} field3_ac_ctx_t;

static void field3_ac(double t, const double r[3], void *ctx, double e[3]) {
    (void)r;
    field3_ac_ctx_t *c = (field3_ac_ctx_t *)ctx;
    e[0] = c->Eamp * sin(c->wx * t);
    e[1] = c->Eamp * sin(c->wy * t);
    e[2] = c->Eamp * sin(c->wz * t);
}

typedef struct {
    double ex, ey, ez;
} field3_dc_ctx_t;

static void field3_dc(double t, const double r[3], void *ctx, double e[3]) {
    (void)t;
    (void)r;
    field3_dc_ctx_t *c = (field3_dc_ctx_t *)ctx;
    e[0] = c->ex;
    e[1] = c->ey;
    e[2] = c->ez;
}

/* 88Sr+ : charge +e, mass ~87.905612 u (neutral 88Sr atom; electron mass negligible). */
static void sr88_plus_init(ion_params3_t *ion3, double fx_hz, double fy_hz, double fz_hz) {
    const double E_CHARGE = 1.602176634e-19;
    const double AMU = 1.66053906660e-27;
    const double M_SR88_U = 87.905612; /* atomic mass of 88Sr in u */
    ion3->q = E_CHARGE;
    ion3->m = M_SR88_U * AMU;
    ion3->omega_sec[0] = 2.0 * M_PI * fx_hz;
    ion3->omega_sec[1] = 2.0 * M_PI * fy_hz;
    ion3->omega_sec[2] = 2.0 * M_PI * fz_hz;
    ion3->axes = NULL;
}

int main(void) {
    spectrum_opts_t opt;
    ion_spectrum_opts_default(&opt);
    opt.dt = 1e-8;
    opt.n = 8192;
    opt.use_hann = 1;
    opt.subtract_mean = 1;

    ion_params3_t ion3;
    sr88_plus_init(&ion3, 1.0e6, 1.2e6, 0.8e6);

    double t0 = 0.0;
    double g3[3];

    printf("88Sr+ demo; secular frequencies 1.0, 1.2, 0.8 MHz (modes 0,1,2)\n\n");

    field3_ac_ctx_t f3 = { .Eamp = 100.0, .wx = ion3.omega_sec[0], .wy = 0.0, .wz = 0.0 };
    ion_heating_rates_from_path_field_3d(t0, &ion3, &opt, path_origin, NULL, field3_ac, &f3,
                                         g3);
    printf("1) E || x at f_x only, Eamp=100 V/m\n");
    printf("   Gamma0 = %.4e, Gamma1 = %.4e, Gamma2 = %.4e 1/s\n",
           g3[0], g3[1], g3[2]);
    printf("   sum = %.4e 1/s\n\n", ion_heating_total_gamma3(g3));

    printf("2) E_amp sweep (x only, on f_x)\n");
    printf("   E_amp [V/m]    Gamma0        Gamma1        Gamma2\n");
    for (int k = 0; k <= 3; k++) {
        f3.Eamp = 20.0 * (1 << k);
        f3.wx = ion3.omega_sec[0];
        f3.wy = 0.0;
        f3.wz = 0.0;
        ion_heating_rates_from_path_field_3d(t0, &ion3, &opt, path_origin, NULL, field3_ac,
                                             &f3, g3);
        printf("   %10.1f    %.4e  %.4e  %.4e\n", f3.Eamp, g3[0], g3[1], g3[2]);
    }
    printf("\n");

    field3_dc_ctx_t d3 = { .ex = 100.0, .ey = -50.0, .ez = 200.0 };
    ion_heating_rates_from_path_field_3d(t0, &ion3, &opt, path_origin, NULL, field3_dc, &d3,
                                         g3);
    printf("3) DC E=(100,-50,200) V/m, mean subtracted\n");
    printf("   Gamma0 = %.3e, Gamma1 = %.3e, Gamma2 = %.3e 1/s\n\n", g3[0], g3[1], g3[2]);

    static const double AX_DIAG[3][3] = {
        {0.5773502691896258, 0.5773502691896258, 0.5773502691896258},
        {0.0, 0.7071067811865476, -0.7071067811865476},
        {0.8164965809277260, -0.4082482904638630, -0.4082482904638630},
    };
    ion3.axes = AX_DIAG;
    f3.Eamp = 100.0;
    f3.wx = ion3.omega_sec[0];
    f3.wy = 0.0;
    f3.wz = 0.0;
    ion_heating_rates_from_path_field_3d(t0, &ion3, &opt, path_origin, NULL, field3_ac, &f3,
                                         g3);
    printf("4) Same as (1) but mode0 along (1,1,1)/sqrt(3): Gamma0 = %.4e 1/s\n", g3[0]);

    return 0;
}
