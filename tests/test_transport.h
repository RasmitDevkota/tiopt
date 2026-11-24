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

int test_parse_parameterized_gates
(
	double atol
);

int test_fast_schedule_matches_baseline
(
	double atol
);

int test_schedule_rounds_fit_zones
(
	double atol
);

int test_parse_qasm_rejects_unknown_gates
(
	double atol
);

int test_schedule_trace_times
(
	double atol
);

int test_trap_graph_routes
(
	double atol
);

int test_schedule_through_junction
(
	double atol
);

int test_three_ion_schedule_by_hand
(
	double atol
);

int test_schedule_check_finds_violations
(
	double atol
);

int test_qccd_cost_model_by_hand
(
	double atol
);

int test_lookahead_layers
(
	double atol
);

int test_commuting_layers
(
	double atol
);

int test_gates_take_chain_in_turn
(
	double atol
);

int test_mid_chain_swap_cost
(
	double atol
);

int test_cycle_breaker_in_storage_zones
(
	double atol
);

int test_stay_evicts_ion_needed_last
(
	double atol
);
