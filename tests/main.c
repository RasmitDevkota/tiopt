#include "main.h"

#include <stdlib.h>
#include <stdio.h>

#include "test_ti_utils.h"
#include "test_electrodynamics.h"
#include "test_io.h"
#include "test_verlet.h"
#include "test_waveforms.h"
#include "test_photonics.h"
#include "test_quantum.h"
#include "test_transport.h"
#include "test_analysis.h"
#include "test_acceleration.h"
#include "test_codesign.h"

// Run every test suite, returning the number of failed tests
int main()
{
	// @TODO - move this into some sort of defs.h
	double default_atol = 1E-9;

	// @TODO - let the user input this somehow
	double atol = default_atol;

	int n_failures = 0;

	printf("--- Breaking into test_ti_utils...\n");
	n_failures += test_all_ti_utils(atol);

	printf("--- Breaking into test_electrodynamics...\n");
	n_failures += test_all_electrodynamics(atol);

	printf("--- Breaking into test_io...\n");
	n_failures += test_all_io(atol);

	printf("--- Breaking into test_verlet...\n");
	n_failures += test_all_verlet(atol);

	printf("--- Breaking into test_waveforms...\n");
	n_failures += test_all_waveforms(atol);

	printf("--- Breaking into test_photonics...\n");
	n_failures += test_all_photonics(atol);

	printf("--- Breaking into test_quantum...\n");
	n_failures += test_all_quantum(atol);

	printf("--- Breaking into test_transport...\n");
	n_failures += test_all_transport(atol);

	printf("--- Breaking into test_analysis...\n");
	n_failures += test_all_analysis(atol);

	printf("--- Breaking into test_acceleration...\n");
	n_failures += test_all_acceleration(atol);

	printf("--- Breaking into test_codesign...\n");
	n_failures += test_all_codesign(atol);

	if (n_failures == 0)
		printf("--- Completed all tests!\n");
	else
		printf("--- Completed all tests with %d failures!\n", n_failures);

	return n_failures > 0;
}
