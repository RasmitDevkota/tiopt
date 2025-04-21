#include "main.h"

#include <stdlib.h>
#include <stdio.h>

// #include <sparselizard.h>
#include <nlopt.h>

#include "data_structures.h"
#include "trap_geometry.h"
#include "electrostatics.h"
// #include "sph.h"

// cost function passed to optimization routine
double cost_function(
  unsigned n,
  const double *x,
  double *grad,
  void *my_func_data
) {
  // the purpose of this function is to compute an aggregate cost of a trap design and transport configuration
  double cost = 0;

  // being lazy with memory for now just as a sketch
  struct Trap trap;

  // generate trap from input parameter array
  generate_trap_from_parameters(&trap, x);

  // generate the initial mesh object needed from the trap electrodes
  generate_mesh(&trap);

  // solve for the potential energy as a 3D array of data for each electrode individually
  solve_trap_electrostatics(&trap);

  // compute the spherical harmonics expansion of the potential energy contribution of every electrode and save in V
  // this can be optimized if we have a saved electrode library that we can recall from if an electrode matches an existing one, just being lazy for now as a sketch
  // expand_spherical_harmonics(trap.V_len, trap.V);

  // perform ion transport experiments now! we don't know how we will actually structure this part, just an example
  for (int e = 0; e < 5; e++)
  {
    // cost_experiment_e = perform_experiment(&trap, e);
    // cost += cost_experiment_e;
  }

  return cost;
}

// main optimization loop
int main()
{
  cost_function(
    0,
    NULL,
    NULL,
    NULL
  );

  double min_cost = 99999;
  const int dim = 1;

  printf("- Constructing optimization problem...\n");

  // nlopt setup
  nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);

  printf("Setting constraints...\n");
  // @TODO - constraints

  printf("Setting bounds...\n");
  // @TODO - bounds

  printf("Setting tolerances...\n");
  // @TODO - tolerances

  printf("Setting objectives...\n");
  nlopt_set_min_objective(opt, cost_function, NULL);

  printf("Constructing initial guess...\n");
  // @TODO - initial guess
  double x[dim];
  for (int i = 0; i < dim; i++)
  {
    x[i] = 0;
  }

  printf("Optimizing...\n");
  if (nlopt_optimize(opt, x, &min_cost) < 0)
  {
    printf("nlopt failed!\n");
  }
  else
  {
    printf("found minimum at f(");
    for (int k = 0; k < dim; k++)
    {
      printf("%g", x[k]);

      if (k < dim-1)
      {
        printf(",");
      }
    }
    printf(") = %0.10g\n", min_cost);
  }

  return 0;
}

