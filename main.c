#include <stdlib.h>

#include <nlopt.h>

struct Electrode
{
  const int n_vertices; // stores a list of the relative coordinates of all the vertices
  const int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
  double (*vertices)[n_vertices][3]; // stores a list of the relative coordinates of all the vertices
  int (*edges)[n_edges][2]; // stores a list of indices of vertices (labelled by index) connected by edges
  const int V_len;
  double (*V)[V_len]; // potential energy per volt in spherical harmonics expansion
  mesh *electrode_mesh;
};

struct Trap
{
  int n_electrodes; // non-constant in case we want to add electrodes to a Trap
  Electrode (*electrodes)[n_electrodes];
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
  Trap trap = {};

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

// basic sketch of how nlopt 
int main()
{
  nlopt_opt opt;
  double *initial_guess;

  // actually set up nlopt stuff...

  if (nlopt_optimize(opt, initial_guess, &cost_function) < 0)
  {
    // nlopt failed!
  }
  else
  {
    // nlopt succeeded!
  }

  return 0;
}
