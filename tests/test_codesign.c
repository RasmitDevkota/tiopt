#include "test_codesign.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "defs.h"
#include "data_structures.h"
#include "codesign.h"
#include "test_utils.h"

// Circuit used by the tests, relative to the tests directory
static char *bell_filename = "../examples/circuits/bell.qasm";

int test_all_codesign
(
	double atol
)
{
	printf("- Testing co-design module...\n");

	int result = 0;

	result += test_create_codesign(atol);
	result += test_codesign_flow(atol);

	return result;
}

int test_create_codesign
(
	double atol
)
{
	struct Codesign *codesign = create_codesign("40Ca+", bell_filename, NULL, NULL);

	if (codesign == NULL)
		return check_error("create_codesign reads Bell circuit", HUGE_VAL, 0.0);

	// The Bell circuit has a Hadamard, a CNOT and a measurement of both qubits, in three layers
	int n_single = 0;
	int n_two = 0;
	int n_measure = 0;
	for (int layer = 0; layer < codesign->n_layers; layer++)
	{
		n_single += codesign->layer_single_qubit_gates[layer];
		n_two += codesign->layer_two_qubit_gates[layer];
		n_measure += codesign->layer_measurements[layer];
	}

	int result = check_error("create_codesign summarizes Bell circuit", abs(codesign->n_ions - 2) + abs(codesign->n_layers - 3) + abs(n_single - 1) + abs(n_two - 1) + abs(n_measure - 2), 0.0);

	// The basic design is a working trap: stable, deep enough, within the voltage limit and with gates
	struct QCMetrics *metrics = &codesign->metrics;
	struct CodesignEnvironment *environment = &codesign->environment;

	int feasible = (
		metrics->radial_frequency > 0.0 &&
		metrics->mathieu_q < environment->max_mathieu_q &&
		metrics->trap_depth > environment->min_trap_depth &&
		metrics->peak_voltage < environment->max_voltage
	);
	result += check_error("basic design is feasible", !feasible, 0.0);

	int sane = (
		metrics->circuit_fidelity > 0.5 && metrics->circuit_fidelity < 1.0 &&
		metrics->two_qubit_fidelity > 0.5 && metrics->two_qubit_fidelity < 1.0 &&
		metrics->single_qubit_fidelity > metrics->two_qubit_fidelity &&
		metrics->detection_fidelity > 0.5 && metrics->detection_fidelity < 1.0 &&
		metrics->circuit_duration > 0.0
	);
	result += check_error("basic design has sane metrics", !sane, 0.0);

	// Without errors the objective is -log(circuit fidelity)
	result += check_error("QC-aware objective of feasible design", fabs(metrics->objective + log(metrics->circuit_fidelity)), atol);

	free_codesign(codesign);

	return result;
}

int test_codesign_flow
(
	double atol
)
{
	struct Codesign *codesign = create_codesign("40Ca+", bell_filename, NULL, NULL);

	if (codesign == NULL)
		return check_error("create_codesign reads Bell circuit", HUGE_VAL, 0.0);

	// A QC-aware stage starts from the current design, so it cannot make the objective worse
	double objective_before = codesign->metrics.objective;

	double objective;
	int n_evaluations;
	optimize_codesign_stage(codesign, PHOTONICS_STAGE, 1, 6, &objective, &n_evaluations);

	int result = check_error("QC-aware stage does not worsen objective", MAX(codesign->metrics.objective - objective_before, 0.0), 1E-12);
	result += check_error("QC-aware stage respects evaluation limit", n_evaluations > 6, 0.0);

	// One pass records every stage after the basic design
	int n_records = codesign->n_records;
	run_codesign(codesign, 1, 3);
	result += check_error("run_codesign records every stage", abs(codesign->n_records - n_records - (N_CODESIGN_STAGES - 1)), 0.0);

	struct CodesignRecord *last = &codesign->records[codesign->n_records - 1];
	result += check_error("run_codesign ends with metrics", (last->stage != METRICS_STAGE) + fabs(last->metrics.circuit_fidelity - codesign->metrics.circuit_fidelity), 0.0);

	free_codesign(codesign);

	return result;
}
