#pragma once

#include "defs.h"

struct Electrode
{
  int n_vertices; // stores a list of the relative coordinates of all the vertices
  int n_edges; // stores a list of indices of vertices connected by edges, labelled by index
  double (*vertices)[3]; // stores a list of the relative coordinates of all the vertices; implicity, length n_vertices-by-3
  int (*edges)[2]; // stores a list of indices of vertices (labelled by index) connected by edges; implicity, length n_edges-by-2
  int Vlm_len; // must equal NSPH_X*NSPH_Y*NSPH_Z*(LMAX+1)*(LMAX+1)*2
  double (*Vlm)[]; // electric potential in spherical harmonics expansion; implicity, length Vlm_len in each component
};

struct LaserPositions
{
       double *laser_frequencies;
       double positions[n_laser_frequencies][][3];
};

struct Trap
{
  int n_electrodes; // non-constant in case we want to add electrodes to a Trap
  int n_electrodes_rf; // non-constant in case we want to add electrodes to a Trap
  int n_electrodes_dc; // non-constant in case we want to add electrodes to a Trap
  struct Electrode (*electrodes)[]; // implicitly, length n_electrodes
  double (*electrode_positions)[][3]; // stores a list of the coordinates of all the electrodes; implicity, length n_electrodes-by-3
  struct LaserPositions laser_positions;
};

struct PulseProgram
{
	double *n_qubits;
	double *laser_pulse_frequencies;
	double *laser_pulse_times;
	double *laser_pulse_powers;
	double *laser_pulse_durations;
	double *laser_pulse_targets;
};


struct TransportProgram 
{
  double *arrival_times;
  int *target_qubits;
  double *destinations[3];
};
