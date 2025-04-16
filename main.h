int main();

// cost function passed to optimization routine
double cost_function(
  unsigned n,
  const double *x,
  double *grad,
  void *my_func_data
);

