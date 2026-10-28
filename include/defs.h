// Constants
#define PI 3.14159265358979
#define SQRT2 1.4142135624
#define INVSQRT2 0.7071067812

// Physical constants (SI units)
#define E_CHARGE 1.602176634E-19
#define AMU 1.66053906660E-27
#define HBAR 1.054571817E-34
#define H_PLANCK 6.62607015E-34
#define C_LIGHT 299792458.0
#define EPSILON_0 8.8541878128E-12
#define K_B 1.380649E-23

// Lengths in the trap frame are measured in micrometers; this converts them to meters
#ifndef LENGTH_SCALE
#define LENGTH_SCALE 1E-6
#endif

#ifndef ION_HEIGHT
#define ION_HEIGHT 55.0
#endif
#ifndef ION_HEIGHT_INT
#define ION_HEIGHT_INT 55
#endif

#ifndef SPH_R
#define SPH_R 20.0
#endif
#ifndef SPH_R_INT
#define SPH_R_INT 20
#endif
#ifndef SPH_SPACING
#define SPH_SPACING (SPH_R * SQRT2)
#endif
#ifndef NSPH_X
#define NSPH_X 16
#endif
#ifndef NSPH_Y
#define NSPH_Y 16
#endif
#ifndef NSPH_Z
#define NSPH_Z 2
#endif
#ifndef SPH_Z_MIN
#define SPH_Z_MIN (ION_HEIGHT - SPH_R * INVSQRT2 * (NSPH_Z - 1 + SQRT2))
#endif

#ifndef RELAXATION_RESOLUTION
#define RELAXATION_RESOLUTION 1
#endif
#ifndef RELAXATION_NX
#define RELAXATION_NX (NSPH_X * (SPH_R_INT + SPH_R_INT / 2))
#endif
#ifndef RELAXATION_NY
#define RELAXATION_NY (NSPH_Y * (SPH_R_INT + SPH_R_INT / 2))
#endif
#ifndef RELAXATION_NZ
#define RELAXATION_NZ (ION_HEIGHT_INT + SPH_R_INT * (NSPH_Z - 1))
#endif
#ifndef RELAXATION_TOLERANCE
#define RELAXATION_TOLERANCE 1E-7
#endif

// The trap frame has its origin at the center of the relaxation grid's bottom face (the electrode surface)
#ifndef TRAP_X0
#define TRAP_X0 (RELAXATION_NX / 2)
#endif
#ifndef TRAP_Y0
#define TRAP_Y0 (RELAXATION_NY / 2)
#endif

// Centers of the spheres used for the spherical harmonics expansions, in the trap frame
#ifndef SPH_CENTER_X
#define SPH_CENTER_X(sx) ((sx) * SPH_SPACING + SPH_R - TRAP_X0)
#endif
#ifndef SPH_CENTER_Y
#define SPH_CENTER_Y(sy) ((sy) * SPH_SPACING + SPH_R - TRAP_Y0)
#endif
#ifndef SPH_CENTER_Z
#define SPH_CENTER_Z(sz) ((sz) * SPH_SPACING + SPH_Z_MIN + 1) // # @TEST - check the +1
#endif

#ifndef LMAX
#define LMAX 10
#endif
// Driscoll-Healy sampling is only exact up to degree NLAT/2 - 1, so we need NLAT = NLON = 2 * (LMAX + 1)
#ifndef NLAT
#define NLAT (2 * (LMAX + 1))
#endif
#ifndef NLON
#define NLON (2 * (LMAX + 1))
#endif

// Number of regular solid harmonics R_l^m with 0 <= m <= l <= LMAX
#ifndef SOLID_HARMONICS_LEN
#define SOLID_HARMONICS_LEN ((LMAX + 1) * (LMAX + 2) / 2)
#endif

// Waveform defaults
#ifndef AXIAL_FREQUENCY
#define AXIAL_FREQUENCY (2 * PI * 1E6) // target axial secular frequency (rad/s)
#endif
#ifndef TRANSPORT_SPEED_LIMIT
#define TRANSPORT_SPEED_LIMIT 1E6 // maximum speed at which wells are moved (µm/s)
#endif
#ifndef WAVEFORM_VOLTAGE_WEIGHT
#define WAVEFORM_VOLTAGE_WEIGHT 1.0
#endif
#ifndef WAVEFORM_ROUGHNESS_WEIGHT
#define WAVEFORM_ROUGHNESS_WEIGHT 1E3
#endif
#ifndef WAVEFORM_CONSTRAINT_WEIGHT
#define WAVEFORM_CONSTRAINT_WEIGHT 1E10
#endif
#ifndef WAVEFORM_CONSTRAINT_TOLERANCE
#define WAVEFORM_CONSTRAINT_TOLERANCE 1E-6
#endif

// Circuit scheduling defaults (s)
#ifndef GATE_DURATION
#define GATE_DURATION 1E-4
#endif
#ifndef DETECTION_DURATION
#define DETECTION_DURATION 2E-4
#endif
#ifndef COOLING_DURATION
#define COOLING_DURATION 5E-4
#endif

// Quantum dynamics defaults
#ifndef QUANTUM_STEP_FACTOR
#define QUANTUM_STEP_FACTOR 0.02 // the first adaptive step is this fraction of the fastest timescale
#endif
#ifndef QUANTUM_TOLERANCE
#define QUANTUM_TOLERANCE 1E-10 // local error tolerance of the adaptive integrator
#endif

// Macros
#ifndef MAX
#define MAX(a, b) \
	({ \
		__typeof__(a) _a = (a); \
		__typeof__(b) _b = (b); \
		_a > _b ? _a : _b; \
	})
#endif

#ifndef MIN
#define MIN(a, b) \
	({ \
		__typeof__(a) _a = (a); \
		__typeof__(b) _b = (b); \
		_a < _b ? _a : _b; \
	})
#endif

#ifndef VLM_SPHERE_LEN
#define VLM_SPHERE_LEN ((LMAX + 1) * (LMAX + 1) * 2)
#endif

#ifndef VLM_SLICE
#define VLM_SLICE(Vlm, sx, sy, sz) \
	(&(*Vlm)[((sx) * NSPH_Y * NSPH_Z + (sy) * NSPH_Z + (sz)) * ((LMAX + 1) * (LMAX + 1) * 2)])
#endif

// Index of the cosine (c = 0) or sine (c = 1) coefficient of degree l and order m within the coefficients of
// one sphere, which follow the Fortran (2, LMAX+1, LMAX+1) column-major layout produced by SHTOOLS
#ifndef VLM_INDEX
#define VLM_INDEX(c, l, m) ((c) + 2 * (l) + 2 * (LMAX + 1) * (m))
#endif
