#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "io.h"
#include "ions.h"
#include "electrodynamics.h"
#include "photonics.h"
#include "quantum.h"
#include "traps.h"
#include "waveforms.h"
#include "codesign.h"
#include "acceleration.h"

// Load the example trap with integrated photonics, look at the ion's confinement and the light it receives at the gate
// zone, and then co-design a trap for a Bell pair circuit
int main
(
	int argc,
	char **argv
)
{
	// Accelerators are optional: OMP_NUM_THREADS sets the number of threads, and OpenCL is used if it is asked for
	if (argc > 1 && atoi(argv[1]) > 0)
		set_num_threads(atoi(argv[1]));

	printf("Running on %d threads\n", get_num_threads());

	struct IonParameters ion;
	generate_ion_from_preset("40Ca+", &ion);

	double m = ion_mass(&ion);
	double q = ion_charge(&ion);

	struct Trap trap;
	if (generate_trap_from_file("../traps/linear_trap_photonics.trap", &trap) != 0)
		return 1;

	solve_trap_electrodynamics(&trap, ANALYTICAL, NULL);

	// RF null above the trap center, and a 1 MHz well there
	double gate_position[3] = { 0.0, 0.0, ION_HEIGHT };
	find_rf_null(&trap, &gate_position, 100);

	double omega_axial = 2 * PI * 1E6;
	double voltages[trap.n_electrodes];
	double well_position[1][3] = { { gate_position[0], gate_position[1], gate_position[2] } };
	solve_waveform(&trap, m, q, 1, 1, well_position, omega_axial, 0.0, voltages);

	double omega[3];
	double axes[3][3];
	compute_normal_modes(&trap, m, q, voltages, gate_position, &omega, &axes);

	printf("RF null at (%.2f, %.2f, %.2f) µm\n", gate_position[0], gate_position[1], gate_position[2]);
	printf("Secular frequencies in the well: %.3f, %.3f, %.3f MHz\n", omega[0] / (2 * PI * 1E6), omega[1] / (2 * PI * 1E6), omega[2] / (2 * PI * 1E6));

	// The qubit beam at the gate zone
	double intensity;
	double k_vector[3];
	evaluate_grating_beam(&(*trap.grating_couplers)[QUBIT_BEAM], gate_position, &intensity, &k_vector);

	double rabi_frequency = rabi_frequency_from_intensity(ion.gamma_SD, 2 * PI * C_LIGHT / ion.omega_SD, intensity);
	double axis[3] = { 1.0, 0.0, 0.0 };
	double eta = lamb_dicke_parameter(k_vector, axis, m, omega_axial);

	printf("Qubit beam: Rabi frequency 2 pi x %.1f kHz, axial Lamb-Dicke parameter %.4f\n", rabi_frequency / (2 * PI * 1E3), eta);

	free_trap(&trap);

	// Co-design for a Bell pair: a pass with each stage's own objective, then a QC-aware pass
	struct Codesign *codesign = create_codesign("40Ca+", "../circuits/bell.qasm", NULL, NULL);
	if (codesign == NULL)
		return 1;

	run_codesign(codesign, 2, 8);

	for (int r = 0; r < codesign->n_records; r++)
		printf("Pass %d, %s: circuit fidelity %.6f\n", codesign->records[r].pass, codesign_stage_name(codesign->records[r].stage), codesign->records[r].metrics.circuit_fidelity);

	free_codesign(codesign);

	return 0;
}
