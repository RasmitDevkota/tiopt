// Fundamental constants
#define PI 3.14159265358979

// Codebase constants
#ifndef RELAXATION_RESOLUTION
#define RELAXATION_RESOLUTION 1
#endif
#ifndef RELAXATION_NX
#define RELAXATION_NX 100
#endif
#ifndef RELAXATION_NY
#define RELAXATION_NY 100
#endif
#ifndef RELAXATION_NZ
#define RELAXATION_NZ 80
#endif

#ifndef SPH_R
#define SPH_R 25.0
#endif

#ifndef SPH_Z_MIN
#define SPH_Z_MIN 30.0
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

