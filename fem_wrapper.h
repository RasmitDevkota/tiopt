#include <stdio.h>

// #include <sparselizard.h>

#ifdef __cplusplus
extern "C"
{
#endif

void sparselizard_wrapper(char *mesh_filename, char *output_filename);
void sparselizard_sample_dh1(void *_f, int sample_volume, int nlat, int nlon);

#ifdef __cplusplus
}
#endif

