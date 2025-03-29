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

int main();

// cost function passed to optimization routine
double cost_function(
  unsigned n,
    const double *x,
    double *grad,
    void *my_func_data
);

