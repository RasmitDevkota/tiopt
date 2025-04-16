#pragma once

// #include <sparselizard.h>

struct Electrode
{
  const int n_vertices; // stores a list of the relative coordinates of all the vertices
  const int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
  double (*vertices)[][3]; // stores a list of the relative coordinates of all the vertices; implicity, length n_vertices
  int (*edges)[][2]; // stores a list of indices of vertices (labelled by index) connected by edges; implicity, length n_edges
  const int V_len;
  double (*V)[]; // potential energy per volt in spherical harmonics expansion; implicity, length V_len
  // mesh *electrode_mesh;
  char *electrode_mesh_filename;
};

struct Trap
{
  int n_electrodes; // non-constant in case we want to add electrodes to a Trap
  struct Electrode (*electrodes)[]; // implicitly, length n_electrodes
};

