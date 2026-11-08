#pragma once

int test_all_transport
(
	double atol
);

int test_add_qasm_targets
(
	double atol
);

int test_parse_qasm
(
	double atol
);

int test_compute_circuit_layers
(
	double atol
);

int test_generate_transport_schedule
(
	double atol
);

int test_parse_qasm_rejects_unknown_gates
(
	double atol
);

int test_schedule_rounds_fit_zones
(
	double atol
);
