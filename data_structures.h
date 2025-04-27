#pragma once

struct Electrode
{
  int n_vertices; // stores a list of the relative coordinates of all the vertices
  int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
  double (*vertices)[3]; // stores a list of the relative coordinates of all the vertices; implicity, length n_vertices-by-3
  int (*edges)[2]; // stores a list of indices of vertices (labelled by index) connected by edges; implicity, length n_edges-by-2
  int V_len;
  double (*V)[]; // potential energy per volt in spherical harmonics expansion; implicity, length V_len
  char electrode_mesh_filename[64];
};

struct Trap
{
  int n_electrodes; // non-constant in case we want to add electrodes to a Trap
  int n_electrodes_rf; // non-constant in case we want to add electrodes to a Trap
  int n_electrodes_dc; // non-constant in case we want to add electrodes to a Trap
  struct Electrode (*electrodes)[]; // implicitly, length n_electrodes
  double (*electrode_positions)[][3]; // stores a list of the coordinates of all the electrodes; implicity, length n_electrodes-by-3
};

