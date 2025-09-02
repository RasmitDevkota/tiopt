// Constants
#define PI 3.14159265358979
#define SQRT2 1.4142135624
#define INVSQRT2 0.7071067812

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
#define RELAXATION_NX (NSPH_X * (SPH_R_INT + SPH_R_INT/2))
#endif
#ifndef RELAXATION_NY
#define RELAXATION_NY (NSPH_Y * (SPH_R_INT + SPH_R_INT/2))
#endif
#ifndef RELAXATION_NZ
#define RELAXATION_NZ (ION_HEIGHT_INT + SPH_R_INT * (NSPH_Z - 1))
#endif

#ifndef NLAT
#define NLAT 20
#endif
#ifndef NLON
#define NLON 20
#endif
#ifndef LMAX
#define LMAX 10
#endif

// Macros
#ifndef MAX
#define MAX(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
	_a > _b ? _a : _b; })
#endif

#ifndef MIN
#define MIN(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
	_a < _b ? _a : _b; })
#endif

#ifndef VLM_SLICE
#define VLM_SLICE(Vlm, sx, sy, sz) \
    (&(*Vlm)[((sx) * NSPH_Y * NSPH_Z + (sy) * NSPH_Z + (sz)) * ((LMAX+1)*(LMAX+1)*2)])
#endif

