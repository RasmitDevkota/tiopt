#include "main.h"

#include <stdlib.h>
#include <stdio.h>

#include "test_electrodynamics.h"

// Main optimization loop
int main()
{
	// @TODO - move this into some sort of defs.h
	double default_atol = 1E-9;

	// @TODO - let the user input this somehow
	double atol = default_atol;

	printf("--- Breaking into test_electrodynamics...");
	test_all_electrodynamics(atol);

	printf("--- Completed all tests!");

	return 0;
}

