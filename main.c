#include <stdlib.h>
#include <stdio.h>

#include <sparselizard.h>
#include <nlopt.h>

#include "sph.h"

struct Electrode
{
  const int n_vertices; // stores a list of the relative coordinates of all the vertices
  const int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
  double (*vertices)[][3]; // stores a list of the relative coordinates of all the vertices; implicity, length n_vertices
  int (*edges)[][2]; // stores a list of indices of vertices (labelled by index) connected by edges; implicity, length n_edges
  const int V_len;
  double (*V)[]; // potential energy per volt in spherical harmonics expansion; implicity, length V_len
  mesh *electrode_mesh;
};

struct Trap
{
  int n_electrodes; // non-constant in case we want to add electrodes to a Trap
  struct Electrode (*electrodes)[]; // implicitly, length n_electrodes
  mesh *trap_mesh;
};

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
  struct Trap trap = {};

  // generate trap from input parameter array
  generate_trap_from_parameters(&trap, x);

  // generate the initial mesh object needed from the trap electrodes
  generate_mesh(&trap);

  // solve for the potential energy as a 3D array of data for each electrode individually
  solve_electrostatics(&trap);

  // compute the spherical harmonics expansion of the potential energy contribution of every electrode and save in V
  // this can be optimized if we have a saved electrode library that we can recall from if an electrode matches an existing one, just being lazy for now as a sketch
  expand_spherical_harmonics(&trap);

  // perform ion transport experiments now! we don't know how we will actually structure this part, just an example
  for (int e = 0; e < 5; e++)
  {
    cost_experiment_e = perform_experiment(&trap, e);
    cost += cost_experiment_e;
  }

  return cost;
}

// main optimization loop
int main()
{
  double min_cost;
  const int dim;

  // nlopt setup
  nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);

  // @TODO - constraints

  // @TODO - bounds
  
  // @TODO - tolerances

  nlopt_set_min_objective(opt, cost_function, NULL);

  // @TODO - initial guess
  double *x;

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
