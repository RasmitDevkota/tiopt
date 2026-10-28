"""Python interface to the tiopt library through ctypes.

The structures in this module mirror those in data_structures.h, and their sizes are checked against the compiled
library whenever the module is imported. The classes wrap the C objects so that their memory is managed from Python.
Units follow the C library, i.e. lengths in the trap frame are in micrometers, times are in seconds, frequencies are in
rad/s, masses are in kg, charges are in C, and energies are in J, unless stated otherwise.
"""

import ctypes as C
import ctypes.util
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Loading the library
# ---------------------------------------------------------------------------------------------------------------------


def library_path():
    """Find libtiopt.so, which is at $TIOPT_LIBRARY if it is set and in the root of the repository otherwise."""
    if os.environ.get("TIOPT_LIBRARY"):
        return Path(os.environ["TIOPT_LIBRARY"])
    return Path(__file__).resolve().parents[2] / "libtiopt.so"


def _load_library():
    path = library_path()
    if not path.exists():
        raise ImportError(f"{path} does not exist; build it by running `make` in the root of the repository")
    return C.CDLL(str(path))


_lib = _load_library()
_libc = C.CDLL(ctypes.util.find_library("c"))
_libc.malloc.restype = C.c_void_p
_libc.malloc.argtypes = [C.c_size_t]
_libc.free.restype = None
_libc.free.argtypes = [C.c_void_p]
_libc.fflush.restype = C.c_int
_libc.fflush.argtypes = [C.c_void_p]


def flush_c_output():
    """Flush the buffered standard output of the C library."""
    _libc.fflush(None)


# ---------------------------------------------------------------------------------------------------------------------
# Data structures (data_structures.h)
# ---------------------------------------------------------------------------------------------------------------------

_double_p = C.POINTER(C.c_double)
_int_p = C.POINTER(C.c_int)
_vec3 = C.c_double * 3
_vec3_p = C.POINTER(_vec3)


class IonParameters(C.Structure):
    """Mirror of struct IonParameters in data_structures.h."""
    _fields_ = [
        ("element", C.c_char_p),
        ("atomic_mass", C.c_double),
        ("charge", C.c_double),
        ("omega_SP", C.c_double),
        ("omega_SD", C.c_double),
        ("gamma_SP", C.c_double),
        ("gamma_SD", C.c_double),
    ]


class ZoneNode(C.Structure):
    """Mirror of struct ZoneNode in data_structures.h."""
    _fields_ = [
        ("zone_type", C.c_int),
        ("zone_capacity", C.c_int),
        ("zone_occupancy", C.c_int),
        ("position", _vec3),
    ]


class TransportEdge(C.Structure):
    """Mirror of struct TransportEdge in data_structures.h."""
    _fields_ = [
        ("edge_type", C.c_int),
        ("n_zones", C.c_int),
        ("zone_nodes", C.POINTER(ZoneNode)),
    ]


class TrapGraphStruct(C.Structure):
    """Mirror of struct TrapGraph in data_structures.h."""
    _fields_ = [
        ("n_zones", C.c_int),
        ("n_transport_edges", C.c_int),
        ("zone_nodes", C.POINTER(ZoneNode)),
        ("transport_edges", C.POINTER(TransportEdge)),
    ]


class Electrode(C.Structure):
    """Mirror of struct Electrode in data_structures.h."""
    _fields_ = [
        ("n_vertices", C.c_int),
        ("n_edges", C.c_int),
        ("vertices", _vec3_p),
        ("edges", C.POINTER(C.c_int * 2)),
        ("Vlm_len", C.c_int),
        ("Vlm", _double_p),
    ]


class WaveguideStruct(C.Structure):
    """Mirror of struct Waveguide in data_structures.h."""
    _fields_ = [
        ("wavelength", C.c_double),
        ("width", C.c_double),
        ("thickness", C.c_double),
        ("n_core", C.c_double),
        ("n_cladding", C.c_double),
        ("n_eff", C.c_double),
    ]


class GratingCouplerStruct(C.Structure):
    """Mirror of struct GratingCoupler in data_structures.h."""
    _fields_ = [
        ("waveguide", WaveguideStruct),
        ("position", _vec3),
        ("azimuth", C.c_double),
        ("period", C.c_double),
        ("fill_factor", C.c_double),
        ("etch_depth", C.c_double),
        ("length", C.c_double),
        ("width", C.c_double),
        ("focal_distance", C.c_double),
        ("power", C.c_double),
        ("efficiency", C.c_double),
    ]


class DetectorStruct(C.Structure):
    """Mirror of struct Detector in data_structures.h."""
    _fields_ = [
        ("position", _vec3),
        ("width", C.c_double),
        ("length", C.c_double),
        ("wavelength", C.c_double),
        ("quantum_efficiency", C.c_double),
        ("dark_count_rate", C.c_double),
    ]


class TrapStruct(C.Structure):
    """Mirror of struct Trap in data_structures.h."""
    _fields_ = [
        ("n_electrodes", C.c_int),
        ("n_electrodes_rf", C.c_int),
        ("n_electrodes_dc", C.c_int),
        ("electrodes", C.POINTER(Electrode)),
        ("electrode_positions", _vec3_p),
        ("lasers", C.c_void_p),
        ("V_RF", C.c_double),
        ("Omega_RF", C.c_double),
        ("n_grating_couplers", C.c_int),
        ("grating_couplers", C.POINTER(GratingCouplerStruct)),
        ("n_detectors", C.c_int),
        ("detectors", C.POINTER(DetectorStruct)),
    ]


class InstructionNode(C.Structure):
    """Mirror of struct InstructionNode in data_structures.h."""
    pass


InstructionNode._fields_ = [
    ("operation_type", C.c_int),
    ("operation_duration", C.c_double),
    ("n_target_ions", C.c_int),
    ("target_ions", _int_p),
    ("graphindex", C.c_int),
    ("n_prev_nodes", C.c_int),
    ("prev_nodes", C.POINTER(InstructionNode)),
]


class CircuitGraphStruct(C.Structure):
    """Mirror of struct CircuitGraph in data_structures.h."""
    _fields_ = [
        ("n_ions", C.c_int),
        ("n_instruction_nodes", C.c_int),
        ("instruction_nodes", C.POINTER(InstructionNode)),
        ("layer_idxs", _int_p),
    ]


class TransportProgramStruct(C.Structure):
    """Mirror of struct TransportProgram in data_structures.h."""
    _fields_ = [
        ("n_ions", C.c_int),
        ("n_transports", C.c_int),
        ("arrival_times", _double_p),
        ("transport_n_targets_list", _int_p),
        ("transport_targets_list", C.POINTER(_int_p)),
        ("destinations", _vec3_p),
        ("initial_positions", _vec3_p),
    ]


class PulseProgramStruct(C.Structure):
    """Mirror of struct PulseProgram in data_structures.h."""
    _fields_ = [
        ("n_ions", C.c_int),
        ("n_pulses", C.c_int),
        ("pulse_frequencies", _double_p),
        ("pulse_times", _double_p),
        ("pulse_powers", _double_p),
        ("pulse_durations", _double_p),
        ("pulse_n_targets_list", _int_p),
        ("pulse_targets_list", C.POINTER(_int_p)),
        ("pulse_phases", _double_p),
    ]


class TransportWaveformStruct(C.Structure):
    """Mirror of struct TransportWaveform in data_structures.h."""
    _fields_ = [
        ("n_waveform_steps", C.c_int),
        ("n_electrodes", C.c_int),
        ("voltages", _double_p),
        ("times", _double_p),
        ("n_wells", C.c_int),
        ("well_positions", _vec3_p),
    ]


class MotionalModeStruct(C.Structure):
    """Mirror of struct MotionalMode in data_structures.h."""
    _fields_ = [
        ("omega", C.c_double),
        ("n_max", C.c_int),
        ("n_bar", C.c_double),
        ("alpha", C.c_double * 2),
        ("heating_rate", C.c_double),
    ]


class SpectrumOptions(C.Structure):
    """Mirror of the spectrum options in ion_heating/ion_heating.h."""
    _fields_ = [
        ("dt", C.c_double),
        ("n", C.c_int),
        ("use_hann", C.c_int),
        ("subtract_mean", C.c_int),
    ]


class CodesignParametersStruct(C.Structure):
    """Mirror of struct CodesignParameters in codesign.h."""
    _fields_ = [
        (name, C.c_double)
        for name in (
            "segment_width", "V_RF", "Omega_RF",
            "grating_length", "grating_width", "rabi_fraction", "detector_size", "detection_time",
            "zone_spacing", "shuttle_duration",
            "omega_axial", "log_roughness_weight",
        )
    ]


class CodesignEnvironmentStruct(C.Structure):
    """Mirror of struct CodesignEnvironment in codesign.h."""
    _fields_ = [("n_segments", C.c_int)] + [
        (name, C.c_double)
        for name in (
            "max_voltage", "qubit_power", "detection_power", "field_noise", "field_noise_exponent", "voltage_noise",
            "dephasing_rate", "n_bar_cooled", "position_drift", "detector_efficiency", "dark_count_density",
            "target_radial_frequency", "max_mathieu_q", "min_trap_depth",
        )
    ]


class QCMetricsStruct(C.Structure):
    """Mirror of struct QCMetrics in codesign.h."""
    _fields_ = [
        (name, C.c_double)
        for name in (
            "radial_frequency", "mathieu_q", "trap_depth", "heating_rate", "static_voltage",
            "rabi_frequency", "lamb_dicke_parameter", "crosstalk", "rabi_sensitivity", "detection_fidelity",
            "shuttle_excitation", "position_error", "frequency_error", "peak_voltage",
            "single_qubit_time", "two_qubit_time", "gate_n_bar", "single_qubit_fidelity", "two_qubit_fidelity",
            "circuit_duration", "circuit_fidelity", "objective",
        )
    ]


class CodesignRecordStruct(C.Structure):
    """Mirror of struct CodesignRecord in codesign.h."""
    _fields_ = [
        ("pass_", C.c_int),
        ("stage", C.c_int),
        ("n_evaluations", C.c_int),
        ("objective", C.c_double),
        ("parameters", CodesignParametersStruct),
        ("metrics", QCMetricsStruct),
    ]


# Names of the structures as they appear in the C library, for checking their sizes
STRUCTURES = {
    "IonParameters": IonParameters,
    "ZoneNode": ZoneNode,
    "TransportEdge": TransportEdge,
    "TrapGraph": TrapGraphStruct,
    "Electrode": Electrode,
    "Waveguide": WaveguideStruct,
    "GratingCoupler": GratingCouplerStruct,
    "Trap": TrapStruct,
    "InstructionNode": InstructionNode,
    "CircuitGraph": CircuitGraphStruct,
    "TransportProgram": TransportProgramStruct,
    "PulseProgram": PulseProgramStruct,
    "TransportWaveform": TransportWaveformStruct,
    "MotionalMode": MotionalModeStruct,
    "Detector": DetectorStruct,
    "CodesignParameters": CodesignParametersStruct,
    "CodesignEnvironment": CodesignEnvironmentStruct,
    "QCMetrics": QCMetricsStruct,
    "CodesignRecord": CodesignRecordStruct,
}

# Enumerations
RELAXATION, ANALYTICAL, SPECTRAL = 0, 1, 2
LINEAR_SURFACE_TRAP, LINEAR_SURFACE_TRAP_WITH_GRATINGS, LINEAR_SURFACE_TRAP_WITH_PHOTONICS = 0, 1, 2
CPU_DEVICE, OPENCL_DEVICE = 0, 1

# Grating couplers added by Trap.add_photonics, in order (enum PhotonicsBeam)
PHOTONICS_BEAMS = ("qubit", "detection", "repump", "quench")

# Stages of the co-design flow (enum CodesignStage)
CODESIGN_STAGES = ("basic design", "trap", "photonics", "transport", "waveforms", "metrics")
TE, TM = 0, 1
IDLE, BACK_AND_FORTH = 0, 1

ZONE_TYPES = {
    "EMPTY": 1 << 0,
    "LOAD": 1 << 1,
    "COOL": 1 << 2,
    "GATE": 1 << 3,
    "MEASURE": 1 << 4,
    "STORAGE": 1 << 5,
    "JUNCTION": 1 << 6,
    "PRE_JUNCTION": 1 << 7,
}

OPERATION_TYPES = {1 << 0: "transport", 1 << 1: "cool", 1 << 2: "detect", 1 << 3: "deshelve", 1 << 4: "qubit"}

# ---------------------------------------------------------------------------------------------------------------------
# Function prototypes
# ---------------------------------------------------------------------------------------------------------------------


def _prototype(name, restype, *argtypes):
    function = getattr(_lib, name)
    function.restype = restype
    function.argtypes = list(argtypes)
    return function


_trap_p = C.POINTER(TrapStruct)
_ion_p = C.POINTER(IonParameters)
_electrode_p = C.POINTER(Electrode)
_waveform_p = C.POINTER(TransportWaveformStruct)
_grating_p = C.POINTER(GratingCouplerStruct)
_waveguide_p = C.POINTER(WaveguideStruct)
_c = C.c_double

# bindings.h
_prototype("tiopt_struct_size", C.c_int, C.c_char_p)
_prototype("tiopt_constant", _c, C.c_char_p)
_prototype("evaluate_trap_potential_batch", None, _trap_p, _double_p, C.c_int, _double_p, _double_p, _double_p)
_prototype("evaluate_potential_energy_batch", None, _trap_p, _c, _c, _double_p, C.c_int, _double_p, _double_p)
_prototype("evaluate_electrode_potential_analytical_batch", None, _electrode_p, _double_p, C.c_int, _double_p, _double_p)
_prototype("evaluate_grating_beam_batch", None, _grating_p, C.c_int, _double_p, _double_p)
_codesign_p = C.c_void_p
_prototype("get_codesign_n_records", C.c_int, _codesign_p)
_prototype("get_codesign_record", None, _codesign_p, C.c_int, C.POINTER(CodesignRecordStruct))
_prototype("get_codesign_state", None, _codesign_p, C.POINTER(CodesignParametersStruct), C.POINTER(CodesignEnvironmentStruct), C.POINTER(QCMetricsStruct))
_prototype("set_codesign_state", None, _codesign_p, C.POINTER(CodesignParametersStruct), C.POINTER(CodesignEnvironmentStruct))
_prototype("get_codesign_trap", _trap_p, _codesign_p)
_prototype("get_codesign_layout", None, _codesign_p, _double_p, _double_p, _int_p, _int_p, _int_p)

# acceleration.h
_prototype("set_num_threads", None, C.c_int)
_prototype("get_num_threads", C.c_int)
_prototype("set_compute_device", C.c_int, C.c_int)
_prototype("get_compute_device", C.c_int)
_prototype("opencl_available", C.c_int)
_prototype("opencl_device_name", C.c_char_p)
_prototype("release_opencl", None)

# codesign.h
_prototype("default_codesign_parameters", None, C.POINTER(CodesignParametersStruct))
_prototype("default_codesign_environment", None, C.POINTER(CodesignEnvironmentStruct))
_prototype("create_codesign", _codesign_p, C.c_char_p, C.c_char_p, C.POINTER(CodesignEnvironmentStruct), C.POINTER(CodesignParametersStruct))
_prototype("free_codesign", None, _codesign_p)
_prototype("evaluate_codesign", C.c_int, _codesign_p, C.c_int, C.POINTER(QCMetricsStruct))
_prototype("optimize_codesign_stage", C.c_int, _codesign_p, C.c_int, C.c_int, C.c_int, _double_p, _int_p)
_prototype("run_codesign", C.c_int, _codesign_p, C.c_int, C.c_int)

# electrodynamics.h
_prototype("solve_trap_electrodynamics", None, _trap_p, C.c_int, C.c_char_p)
_prototype("solver_relaxation", C.c_int, C.c_int, C.c_int, C.c_int, _electrode_p, _double_p, _double_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int)
_prototype("evaluate_electrode_potential_analytical", None, _electrode_p, _double_p, _double_p, _double_p)
_prototype("expand_spherical_harmonics", None, _double_p, _double_p)
_prototype("find_nearest_sphere", C.c_int, _double_p, _int_p, _int_p, _int_p)
_prototype("evaluate_electrode_derivatives", C.c_int, _electrode_p, _double_p, _double_p, _double_p, _double_p)
_prototype("evaluate_trap_potential", C.c_int, _trap_p, _double_p, _double_p, _double_p, _double_p, _double_p)
_prototype("evaluate_pseudopotential", C.c_int, _trap_p, _c, _c, _double_p, _double_p, _double_p, _double_p)
_prototype("evaluate_potential_energy", C.c_int, _trap_p, _c, _c, _double_p, _double_p, _double_p, _double_p, _double_p)
_prototype("compute_normal_modes", C.c_int, _trap_p, _c, _c, _double_p, _double_p, _double_p, _double_p)
_prototype("find_potential_minimum", C.c_int, _trap_p, _c, _c, _double_p, _double_p, C.c_int)
_prototype("find_rf_null", C.c_int, _trap_p, _double_p, C.c_int)

# traps.h, io.h, ions.h
_prototype("generate_trap_from_preset", C.c_int, C.c_int, C.c_int, _c, _trap_p)
_prototype("allocate_trap", None, _trap_p, C.c_int)
_prototype("allocate_grating_couplers", None, _trap_p, C.c_int)
_prototype("allocate_detectors", None, _trap_p, C.c_int)
_prototype("add_trap_photonics", C.c_int, _trap_p, _double_p, _double_p)
_prototype("set_polygon_electrode", None, _trap_p, C.c_int, C.c_int, _double_p, _double_p)
_prototype("set_rectangular_electrode", None, _trap_p, C.c_int, _c, _c, _c, _c)
_prototype("free_trap", None, _trap_p)
_prototype("generate_trap_from_file", C.c_int, C.c_char_p, _trap_p)
_prototype("write_trap_to_file", C.c_int, C.c_char_p, _trap_p)
_prototype("generate_ion_from_preset", C.c_int, C.c_char_p, _ion_p)
_prototype("ion_mass", _c, _ion_p)
_prototype("ion_charge", _c, _ion_p)

# waveforms.h
_prototype("allocate_transport_waveform", _waveform_p, C.c_int, C.c_int, C.c_int)
_prototype("free_transport_waveform", None, _waveform_p)
_prototype("solve_waveform", C.c_int, _trap_p, _c, _c, C.c_int, C.c_int, _double_p, _c, _c, _double_p)
_prototype("optimize_primitive", C.c_int, _trap_p, _ion_p, _waveform_p, _c, _c, _c, C.c_int)
_prototype("generate_transport_waveform", _waveform_p, _trap_p, _ion_p, _double_p, _double_p, _c, C.c_int, _c)
_prototype("generate_transport_primitive", _waveform_p, _trap_p, _ion_p, _double_p, _double_p, _c, C.c_int, _c, _c)
_prototype("transport_program_to_waveform", _waveform_p, _trap_p, _ion_p, C.POINTER(TransportProgramStruct), _int_p, _c)
_prototype("evaluate_waveform_errors", None, _trap_p, _ion_p, _waveform_p, _c, _double_p, _double_p)
_prototype("waveform_voltages_at_time", None, _waveform_p, _c, _double_p)
_prototype("interpolate_waveform", None, _trap_p, _waveform_p, _waveform_p, _c)
_prototype("compute_secular_frequencies", None, _c, _c, _c, _c, _c, _c, _double_p)

# verlet.h
_prototype("velocity_verlet", None, _trap_p, _c, _c, _waveform_p, C.c_int, _double_p, _double_p, _double_p, C.c_int, _c, _double_p, _double_p)
_prototype("simulate_ensemble", None, _trap_p, _c, _c, _waveform_p, C.c_int, C.c_int, _double_p, _double_p, C.c_int, _c)

# analysis.h
_prototype("heating_rate", _c, _c, _c, _c, _c, _c, _c, _c)
_prototype("heating_rate_rf_noise", _c, _trap_p, _c, _c, _double_p, _c, _double_p, _c)
_prototype("heating_rate_field_noise", _c, _c, _c, _c, _c)
_prototype("heating_rate_voltage_noise", _c, _trap_p, _c, _c, _double_p, _c, _double_p, _c)
_prototype("motional_quanta", _c, _c, _c, _double_p, _double_p, _double_p, _double_p)
_prototype("compute_trap_depth", _c, _trap_p, _c, _c, _double_p)

# experiments.h
_prototype("run_experiment", None, _trap_p, _ion_p, C.c_int, _double_p)

# photonics.h
_prototype("solve_slab_mode", C.c_int, _c, _c, _c, _c, _c, C.c_int, C.c_int, _double_p)
_prototype("solve_waveguide_mode", C.c_int, _waveguide_p)
_prototype("compute_grating_effective_index", C.c_int, _grating_p, _double_p)
_prototype("compute_grating_emission", C.c_int, _grating_p, _double_p, _double_p)
_prototype("design_grating_coupler", C.c_int, _grating_p, _double_p)
_prototype("compute_gaussian_beam", None, _c, _c, _c, _double_p, _double_p)
_prototype("evaluate_grating_beam", C.c_int, _grating_p, _double_p, _double_p, _double_p)
_prototype("compute_collection_efficiency", _c, C.POINTER(DetectorStruct), _double_p)
_prototype("compute_scattering_rate", _c, _c, _c, _c)
_prototype("compute_detection_fidelity", _c, _c, _c, _c, _c, _int_p)
_prototype("compute_multilayer_response", C.c_int, C.c_int, _double_p, _double_p, _double_p, _c, _c, C.c_int, _double_p, _double_p, _double_p, _double_p)
_prototype("compute_grating_directionality", _c, _c, _c, _c, _c, _c, _c, C.c_int)
_prototype("fit_polynomial", C.c_int, C.c_int, _double_p, _double_p, C.c_int, _double_p, _double_p)
_prototype("fit_gaussian", C.c_int, C.c_int, _double_p, _double_p, _double_p)
_prototype("fit_waveguide_dispersion", C.c_int, _waveguide_p, _c, _c, C.c_int, C.c_int, _double_p, _double_p)

# quantum.h
_prototype("allocate_pulse_program", C.POINTER(PulseProgramStruct), C.c_int, C.c_int)
_prototype("free_pulse_program", None, C.POINTER(PulseProgramStruct))
_prototype("lamb_dicke_parameter", _c, _double_p, _double_p, _c, _c)
_prototype("saturation_intensity", _c, _c, _c)
_prototype("rabi_frequency_from_intensity", _c, _c, _c, _c)
_prototype("displacement_matrix_element", None, C.c_int, C.c_int, _c, _c, _double_p)
_prototype("motional_coupling", None, C.c_int, C.c_int, _c, _double_p)
_prototype("thermal_rabi_excitation", _c, _c, _c, C.c_int, _c, _c, _c, C.c_int)
_prototype("simulate_pulse_program", C.c_int, _ion_p, C.POINTER(MotionalModeStruct), C.POINTER(PulseProgramStruct), _double_p, _double_p, _c, C.c_int, _double_p, _double_p, _double_p, _double_p)

# transport.h, circuits.h
_prototype("parse_qasm", C.POINTER(CircuitGraphStruct), C.c_char_p)
_prototype("generate_transport_schedule", C.POINTER(TransportProgramStruct), C.POINTER(TrapGraphStruct), C.POINTER(CircuitGraphStruct), C.POINTER(TransportProgramStruct))
_prototype("generate_linear_trap_graph", None, C.c_int, _int_p, _int_p, _c, _c, _c, C.POINTER(TrapGraphStruct))
_prototype("free_trap_graph", None, C.POINTER(TrapGraphStruct))
_prototype("free_circuit_graph", None, C.POINTER(CircuitGraphStruct))
_prototype("free_transport_program", None, C.POINTER(TransportProgramStruct))

# ion_heating/ion_heating.h
_prototype("ion_spectrum_opts_default", None, C.POINTER(SpectrumOptions))
_prototype("ion_compute_psd", C.c_int, _double_p, C.POINTER(SpectrumOptions), C.POINTER(_double_p), C.POINTER(_double_p))


def struct_size(name):
    """Get the size of a structure in the C library in bytes."""
    return _lib.tiopt_struct_size(name.encode())


def _check_structures():
    for name, structure in STRUCTURES.items():
        if C.sizeof(structure) != struct_size(name):
            raise ImportError(f"struct {name} has size {struct_size(name)} in libtiopt.so but {C.sizeof(structure)} in the bindings")


_check_structures()


def constant(name):
    """Get the value of a compile-time constant of the C library (see defs.h)."""
    value = _lib.tiopt_constant(name.encode())
    if np.isnan(value):
        raise KeyError(name)
    return value


ION_HEIGHT = constant("ION_HEIGHT")
LENGTH_SCALE = constant("LENGTH_SCALE")
AXIAL_FREQUENCY = constant("AXIAL_FREQUENCY")
TRANSPORT_SPEED_LIMIT = constant("TRANSPORT_SPEED_LIMIT")
E_CHARGE = 1.602176634e-19
HBAR = 1.054571817e-34
C_LIGHT = 299792458.0

# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------


def _array(values, shape=None):
    """Make a contiguous float64 copy of values, optionally reshaped."""
    array = np.ascontiguousarray(np.array(values, dtype=np.float64))
    if shape is not None:
        array = array.reshape(shape)
    return array


def _ptr(array):
    return array.ctypes.data_as(_double_p)


def _from_pointer(pointer, n):
    """Copy n doubles from a C pointer into a new array."""
    if n == 0 or not pointer:
        return np.zeros(0)
    return np.ctypeslib.as_array(pointer, shape=(n,)).copy()


class TiOptError(RuntimeError):
    """Error raised when a function of the C library reports a failure."""


# ---------------------------------------------------------------------------------------------------------------------
# Ions
# ---------------------------------------------------------------------------------------------------------------------


class Ion:
    """Trapped ion species, from the presets in ions.c."""

    PRESETS = ("40Ca+", "88Sr+", "171Yb+", "9Be+")

    def __init__(self, element="40Ca+"):
        self.struct = IonParameters()
        if _lib.generate_ion_from_preset(element.encode(), C.byref(self.struct)) != 0:
            raise ValueError(f"unknown ion species {element!r}; choose from {self.PRESETS}")
        self.element = element

    @property
    def mass(self):
        """Mass of the ion (kg)."""
        return _lib.ion_mass(C.byref(self.struct))

    @property
    def charge(self):
        """Charge of the ion (C)."""
        return _lib.ion_charge(C.byref(self.struct))

    @property
    def qubit_frequency(self):
        """Angular frequency of the S-D qubit transition (rad/s)."""
        return self.struct.omega_SD

    @property
    def qubit_wavelength(self):
        """Vacuum wavelength of the S-D qubit transition (m)."""
        return 2 * np.pi * C_LIGHT / self.struct.omega_SD

    @property
    def qubit_linewidth(self):
        """Natural linewidth of the S-D qubit transition (1/s)."""
        return self.struct.gamma_SD

    def __repr__(self):
        return f"Ion({self.element!r})"


# ---------------------------------------------------------------------------------------------------------------------
# Traps
# ---------------------------------------------------------------------------------------------------------------------


class Trap:
    """Surface trap, including its electrodes, RF drive, integrated photonics, and solved electrodynamics."""

    def __init__(self):
        self.struct = TrapStruct()
        self.solved = False
        self.solver = None

    @classmethod
    def from_preset(cls, n_segments=7, segment_width=50.0, gratings=False, photonics=False):
        """Generate the five-wire linear surface trap preset (see generate_trap_from_preset).

        The RF null of the preset is at ION_HEIGHT above the center of the trap.

        Args:
            n_segments: Number of DC segments on each side of the trap.
            segment_width: Width of each DC segment along the trap axis (µm).
            gratings: Specifies whether to add a 729 nm grating coupler which addresses the center of the trap.
            photonics: Specifies whether to add the full integrated photonics of Trap.add_photonics, with the gate zone
                at the center of the trap and the measurement zone two segments along.

        Returns:
            New trap
        """
        trap = cls()
        preset = LINEAR_SURFACE_TRAP
        if gratings:
            preset = LINEAR_SURFACE_TRAP_WITH_GRATINGS
        if photonics:
            preset = LINEAR_SURFACE_TRAP_WITH_PHOTONICS
        if _lib.generate_trap_from_preset(preset, int(n_segments), float(segment_width), C.byref(trap.struct)) != 0:
            raise TiOptError("failed to generate trap from preset")
        return trap

    @classmethod
    def from_file(cls, path):
        """Read a trap from a trap file (see trap.data for the format)."""
        trap = cls()
        if _lib.generate_trap_from_file(str(path).encode(), C.byref(trap.struct)) != 0:
            _lib.free_trap(C.byref(trap.struct))
            raise TiOptError(f"failed to read trap file {path}")
        return trap

    @classmethod
    def from_polygons(cls, polygons, n_rf, V_RF=0.0, Omega_RF=0.0):
        """Build a trap from polygonal electrodes.

        Args:
            polygons: List of (n_vertices, 2 or 3) arrays of the absolute coordinates of the vertices of each electrode
                (µm).
            n_rf: Number of RF electrodes, which are the first n_rf polygons.
            V_RF: Amplitude of the RF voltage (V).
            Omega_RF: Angular frequency of the RF drive (rad/s).

        Returns:
            New trap
        """
        trap = cls()
        _lib.allocate_trap(C.byref(trap.struct), len(polygons))
        trap.struct.n_electrodes_rf = n_rf
        trap.struct.n_electrodes_dc = len(polygons) - n_rf
        trap.struct.V_RF = V_RF
        trap.struct.Omega_RF = Omega_RF
        for e, polygon in enumerate(polygons):
            polygon = np.asarray(polygon, dtype=float)
            if polygon.shape[1] == 2:
                polygon = np.column_stack([polygon, np.zeros(len(polygon))])
            center = polygon.mean(axis=0)
            vertices = _array(polygon - center)
            _lib.set_polygon_electrode(C.byref(trap.struct), e, len(polygon), _ptr(vertices), _ptr(_array(center)))
        return trap

    def save(self, path):
        """Write the trap to a trap file."""
        if _lib.write_trap_to_file(str(path).encode(), C.byref(self.struct)) != 0:
            raise TiOptError(f"failed to write trap file {path}")

    def solve(self, solver="analytical", grid_filename=None):
        """Solve for the potential of every electrode and expand it in spherical harmonics.

        The analytical solver uses the gapless plane approximation. The relaxation solver solves Laplace's equation on a
        grid inside a grounded box, and the spectral solver solves the same discrete problem exactly with fast sine
        transforms (for electrodes in the trap surface, and otherwise it falls back to relaxation).

        Args:
            solver: Solver to use, which is "analytical", "relaxation", or "spectral".
            grid_filename: File to write the potential grids of the grid solvers to, if any.
        """
        solvers = {"analytical": ANALYTICAL, "relaxation": RELAXATION, "spectral": SPECTRAL}
        if solver not in solvers:
            raise ValueError(f"unknown solver {solver!r}")
        filename = str(grid_filename).encode() if grid_filename is not None else None
        _lib.solve_trap_electrodynamics(C.byref(self.struct), solvers[solver], filename)
        flush_c_output()
        self.solved = True
        self.solver = solver

    def close(self):
        """Free the memory of the trap."""
        if self.struct.electrodes:
            _lib.free_trap(C.byref(self.struct))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # Properties ---------------------------------------------------------------------------------------------------

    @property
    def n_electrodes(self):
        """Number of electrodes."""
        return self.struct.n_electrodes

    @property
    def n_rf(self):
        """Number of RF electrodes."""
        return self.struct.n_electrodes_rf

    @property
    def n_dc(self):
        """Number of DC electrodes."""
        return self.struct.n_electrodes_dc

    @property
    def V_RF(self):
        """Amplitude of the RF voltage (V)."""
        return self.struct.V_RF

    @V_RF.setter
    def V_RF(self, value):
        self.struct.V_RF = float(value)

    @property
    def Omega_RF(self):
        """Angular frequency of the RF drive (rad/s)."""
        return self.struct.Omega_RF

    @Omega_RF.setter
    def Omega_RF(self, value):
        self.struct.Omega_RF = float(value)

    def electrode_positions(self):
        """Get the position of every electrode (µm)."""
        return np.array([list(self.struct.electrode_positions[e]) for e in range(self.n_electrodes)]).reshape(-1, 3)

    def electrode_polygons(self):
        """Get the absolute coordinates of the vertices of every electrode as (n_vertices, 3) arrays."""
        polygons = []
        positions = self.electrode_positions()
        for e in range(self.n_electrodes):
            electrode = self.struct.electrodes[e]
            vertices = np.array([list(electrode.vertices[v]) for v in range(electrode.n_vertices)]).reshape(-1, 3)
            polygons.append(vertices + positions[e])
        return polygons

    def is_rf(self, e):
        """Check whether electrode e is an RF electrode."""
        return e < self.n_rf

    def grating_couplers(self):
        """Get the grating couplers of the trap as Grating objects, which share memory with the trap."""
        return [Grating(self.struct.grating_couplers[g]) for g in range(self.struct.n_grating_couplers)]

    def detectors(self):
        """Get the integrated detectors of the trap as Detector objects, which share memory with the trap."""
        return [Detector(self.struct.detectors[d]) for d in range(self.struct.n_detectors)]

    def add_photonics(self, gate_position, measure_position):
        """Replace the photonics of the trap with a grating coupler for every 40Ca+ beam and a detector.

        The beams are given by PHOTONICS_BEAMS.

        Args:
            gate_position: Position of the gate zone (µm), which the qubit and quench beams are focused on.
            measure_position: Position of the measurement zone (µm), which the detection and repump beams are focused on
                and the detector is placed beneath.
        """
        if _lib.add_trap_photonics(C.byref(self.struct), _ptr(_array(gate_position)), _ptr(_array(measure_position))) != 0:
            raise TiOptError("failed to design the trap's grating couplers")

    def zero_voltages(self):
        """Get a voltage of 0 V for every electrode."""
        return np.zeros(self.n_electrodes)

    def _voltages(self, voltages):
        if voltages is None:
            return self.zero_voltages()
        voltages = _array(voltages)
        if voltages.shape != (self.n_electrodes,):
            raise ValueError(f"expected {self.n_electrodes} voltages, got {voltages.shape}")
        return voltages

    def _require_solved(self):
        if not self.solved:
            raise TiOptError("the trap's electrodynamics must be solved first (Trap.solve)")

    def rf_weights(self):
        """Get the voltages which drive every RF electrode at 1 V, for evaluating the RF potential."""
        weights = self.zero_voltages()
        weights[: self.n_rf] = 1.0
        return weights

    # Electrodynamics ----------------------------------------------------------------------------------------------

    def potential(self, positions, voltages=None, gradient=False):
        """Evaluate the potential at many positions at once (on the OpenCL device if it is selected).

        Args:
            positions: Positions (µm) with shape (..., 3).
            voltages: Voltage of each electrode (V), where None means 0 V on every electrode.
            gradient: Specifies whether to also return the gradient of the potential.

        Returns:
            Potential (V) with shape (...), and its gradient (V/µm) with shape (..., 3) if it is requested
        """
        self._require_solved()
        positions = _array(positions)
        shape = positions.shape[:-1]
        flat = _array(positions.reshape(-1, 3))
        phi = np.zeros(len(flat))
        grad = np.zeros((len(flat), 3)) if gradient else None
        _lib.evaluate_trap_potential_batch(
            C.byref(self.struct), _ptr(self._voltages(voltages)), len(flat), _ptr(flat), _ptr(phi), _ptr(grad) if gradient else None
        )
        if gradient:
            return phi.reshape(shape), grad.reshape(shape + (3,))
        return phi.reshape(shape)

    def potential_energy(self, ion, positions, voltages=None):
        """Evaluate the potential energy of an ion at many positions at once, including the RF pseudopotential.

        Args:
            ion: Ion species.
            positions: Positions (µm) with shape (..., 3).
            voltages: Voltage of each electrode (V), where None means 0 V on every electrode.

        Returns:
            Potential energy (J) with shape (...)
        """
        self._require_solved()
        positions = _array(positions)
        shape = positions.shape[:-1]
        flat = _array(positions.reshape(-1, 3))
        U = np.zeros(len(flat))
        _lib.evaluate_potential_energy_batch(
            C.byref(self.struct), ion.mass, ion.charge, _ptr(self._voltages(voltages)), len(flat), _ptr(flat), _ptr(U)
        )
        return U.reshape(shape)

    def pseudopotential(self, ion, positions):
        """Evaluate the RF pseudopotential (J) of an ion at positions with shape (..., 3)."""
        return self.potential_energy(ion, positions, None)

    def analytical_electrode_potential(self, e, positions):
        """Evaluate the gapless plane potential (V/V) of electrode e at positions with shape (..., 3), without the
        spherical harmonic expansion.
        """
        positions = _array(positions)
        shape = positions.shape[:-1]
        flat = _array(positions.reshape(-1, 3))
        phi = np.zeros(len(flat))
        position = _array(self.struct.electrode_positions[e])
        _lib.evaluate_electrode_potential_analytical_batch(
            C.byref(self.struct.electrodes[e]), _ptr(position), len(flat), _ptr(flat), _ptr(phi)
        )
        return phi.reshape(shape)

    def evaluate(self, position, voltages=None):
        """Evaluate the potential (V), its gradient (V/µm), and its Hessian (V/µm^2) at one position."""
        self._require_solved()
        phi = C.c_double()
        grad = np.zeros(3)
        H = np.zeros((3, 3))
        _lib.evaluate_trap_potential(C.byref(self.struct), _ptr(self._voltages(voltages)), _ptr(_array(position)), C.byref(phi), _ptr(grad), _ptr(H))
        return phi.value, grad, H

    def energy_derivatives(self, ion, position, voltages=None):
        """Evaluate the potential energy (J) of an ion, its gradient (J/µm), and its Hessian (J/µm^2) at one position.
        """
        self._require_solved()
        U = C.c_double()
        grad = np.zeros(3)
        H = np.zeros((3, 3))
        _lib.evaluate_potential_energy(
            C.byref(self.struct), ion.mass, ion.charge, _ptr(self._voltages(voltages)), _ptr(_array(position)), C.byref(U), _ptr(grad), _ptr(H)
        )
        return U.value, grad, H

    def normal_modes(self, ion, position, voltages=None):
        """Compute the secular frequencies (rad/s, negative for unstable directions) and principal axes (as rows) at a
        position.
        """
        self._require_solved()
        omega = np.zeros(3)
        axes = np.zeros((3, 3))
        _lib.compute_normal_modes(C.byref(self.struct), ion.mass, ion.charge, _ptr(self._voltages(voltages)), _ptr(_array(position)), _ptr(omega), _ptr(axes))
        return omega, axes

    def find_rf_null(self, position=(0.0, 0.0, None), max_iterations=100):
        """Find the point of the RF null line at the x-coordinate of position, starting the search at ION_HEIGHT if z is
        None.
        """
        self._require_solved()
        x, y, z = position
        point = _array([x, y, ION_HEIGHT if z is None else z])
        if _lib.find_rf_null(C.byref(self.struct), _ptr(point), max_iterations) != 0:
            raise TiOptError("failed to find RF null")
        return point

    def find_minimum(self, ion, position, voltages=None, max_iterations=100):
        """Find the stationary point of the potential energy nearest to position."""
        self._require_solved()
        point = _array(position)
        status = _lib.find_potential_minimum(C.byref(self.struct), ion.mass, ion.charge, _ptr(self._voltages(voltages)), _ptr(point), max_iterations)
        if status != 0:
            raise TiOptError("failed to find a stationary point of the potential energy")
        return point

    # Analysis -----------------------------------------------------------------------------------------------------

    def trap_depth(self, ion, rf_null):
        """Compute the depth of the pseudopotential at an RF null (eV), from the barrier along the vertical through it.
        """
        return _lib.compute_trap_depth(C.byref(self.struct), ion.mass, ion.charge, _ptr(_array(rf_null)))

    def voltage_noise_heating_rate(self, ion, position, omega, axis=(1.0, 0.0, 0.0), S_V=1e-18):
        """Compute the heating rate (quanta/s) from uncorrelated voltage noise S_V (V^2/Hz) on every DC electrode."""
        self._require_solved()
        return _lib.heating_rate_voltage_noise(C.byref(self.struct), ion.mass, ion.charge, _ptr(_array(position)), omega, _ptr(_array(axis)), S_V)

    def rf_noise_heating_rate(self, ion, position, omega, axis=(0.0, 0.0, 1.0), S_V_N=1e-12):
        """Compute the heating rate (quanta/s) from noise S_V_N (V^2/Hz) on the amplitude of the RF drive."""
        self._require_solved()
        return _lib.heating_rate_rf_noise(C.byref(self.struct), ion.mass, ion.charge, _ptr(_array(position)), omega, _ptr(_array(axis)), S_V_N)

    def run_experiment(self, ion, experiment="idle"):
        """Run one of the experiments in experiments.c ("idle" or "back_and_forth") and return its cost."""
        self._require_solved()
        experiments = {"idle": IDLE, "back_and_forth": BACK_AND_FORTH}
        cost = C.c_double(0.0)
        _lib.run_experiment(C.byref(self.struct), C.byref(ion.struct), experiments[experiment], C.byref(cost))
        flush_c_output()
        return cost.value


# ---------------------------------------------------------------------------------------------------------------------
# Waveforms and dynamics
# ---------------------------------------------------------------------------------------------------------------------


class Waveform:
    """Electrode voltages over time, which move potential wells through a trap."""

    def __init__(self, pointer):
        if not pointer:
            raise TiOptError("failed to generate waveform")
        self._pointer = pointer

    @classmethod
    def empty(cls, n_steps, n_electrodes, n_wells=1):
        """Allocate a waveform with n_steps steps, n_electrodes electrodes, and n_wells wells."""
        return cls(_lib.allocate_transport_waveform(n_steps, n_electrodes, n_wells))

    @classmethod
    def transport(cls, trap, ion, start, end, duration, n_steps, omega_axial=AXIAL_FREQUENCY):
        """Solve for a waveform which moves a well between two positions along a minimum-jerk profile.

        Args:
            trap: Trap whose electrodynamics have been solved.
            ion: Ion species.
            start: Starting position of the well (µm).
            end: Final position of the well (µm).
            duration: Duration of the transport (s), where 0 gives a static well.
            n_steps: Number of steps of the waveform.
            omega_axial: Axial secular frequency of the well (rad/s).

        Returns:
            New waveform
        """
        trap._require_solved()
        pointer = _lib.generate_transport_waveform(
            C.byref(trap.struct), C.byref(ion.struct), _ptr(_array(start)), _ptr(_array(end)), duration, n_steps, omega_axial
        )
        flush_c_output()
        return cls(pointer)

    @classmethod
    def primitive(cls, trap, ion, start, end, duration, n_steps, omega_axial=AXIAL_FREQUENCY, roughness_weight=1e3):
        """Solve for a transport like Waveform.transport, with the given weight on the roughness of the voltages between
        steps.
        """
        trap._require_solved()
        pointer = _lib.generate_transport_primitive(
            C.byref(trap.struct), C.byref(ion.struct), _ptr(_array(start)), _ptr(_array(end)), duration, n_steps, omega_axial, roughness_weight
        )
        flush_c_output()
        return cls(pointer)

    @classmethod
    def from_program(cls, trap, ion, program, n_steps_per_transport=101, omega_axial=AXIAL_FREQUENCY):
        """Solve for wells for every ion which are moved by each transport of a TransportProgram."""
        trap._require_solved()
        steps = (C.c_int * program.n_transports)(*([n_steps_per_transport] * program.n_transports))
        pointer = _lib.transport_program_to_waveform(C.byref(trap.struct), C.byref(ion.struct), program._pointer, steps, omega_axial)
        flush_c_output()
        return cls(pointer)

    def close(self):
        """Free the memory of the waveform."""
        if self._pointer:
            _lib.free_transport_waveform(self._pointer)
            self._pointer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def struct(self):
        """Underlying struct TransportWaveform."""
        return self._pointer.contents

    @property
    def n_steps(self):
        """Number of steps."""
        return self.struct.n_waveform_steps

    @property
    def n_electrodes(self):
        """Number of electrodes."""
        return self.struct.n_electrodes

    @property
    def n_wells(self):
        """Number of potential wells."""
        return self.struct.n_wells

    @property
    def times(self):
        """Time of each step (s)."""
        return _from_pointer(self.struct.times, self.n_steps)

    @property
    def voltages(self):
        """Voltages (V) with shape (n_steps, n_electrodes)."""
        return _from_pointer(self.struct.voltages, self.n_steps * self.n_electrodes).reshape(self.n_steps, self.n_electrodes)

    @property
    def well_positions(self):
        """Target positions of the wells with shape (n_steps, n_wells, 3)."""
        n = self.n_steps * self.n_wells * 3
        flat = _from_pointer(C.cast(self.struct.well_positions, _double_p), n)
        return flat.reshape(self.n_steps, self.n_wells, 3)

    @property
    def duration(self):
        """Duration of the waveform (s)."""
        times = self.times
        return times[-1] - times[0] if len(times) else 0.0

    def voltages_at(self, t):
        """Interpolate the voltages (V) at time t (s)."""
        voltages = np.zeros(self.n_electrodes)
        _lib.waveform_voltages_at_time(self._pointer, t, _ptr(voltages))
        return voltages

    def errors(self, trap, ion, omega_axial=AXIAL_FREQUENCY):
        """Compute the largest residual force (N) on an ion at any well and the largest relative error of the axial
        frequency.
        """
        max_force = C.c_double()
        max_frequency_error = C.c_double()
        _lib.evaluate_waveform_errors(C.byref(trap.struct), C.byref(ion.struct), self._pointer, omega_axial, C.byref(max_force), C.byref(max_frequency_error))
        return max_force.value, max_frequency_error.value

    def optimize(self, trap, ion, omega_axial=AXIAL_FREQUENCY, max_voltage=10.0, max_total_voltage=0.0, max_iterations=500):
        """Refine the waveform with NLopt, enforcing the well constraints and voltage limits (which is slow for long
        waveforms).
        """
        status = _lib.optimize_primitive(C.byref(trap.struct), C.byref(ion.struct), self._pointer, omega_axial, max_voltage, max_total_voltage, max_iterations)
        flush_c_output()
        if status != 0:
            raise TiOptError("NLopt failed to refine the waveform")

    def resampled(self, trap, n_steps):
        """Interpolate the waveform linearly onto n_steps evenly spaced steps."""
        fine = Waveform.empty(n_steps, self.n_electrodes, self.n_wells)
        dx = (self.n_steps - 1) / (n_steps - 1) if n_steps > 1 else 0.0
        _lib.interpolate_waveform(C.byref(trap.struct), self._pointer, fine._pointer, dx)
        return fine


@dataclass
class Trajectory:
    """Trajectory of an ion, with its positions (µm) and velocities (µm/s) at each time (s)."""
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray


def simulate_trajectory(trap, ion, waveform, duration, dt, initial_position, initial_velocity=(0.0, 0.0, 0.0), full_rf=False):
    """Integrate the motion of an ion with velocity Verlet while a waveform is applied.

    Args:
        trap: Trap whose electrodynamics have been solved.
        ion: Ion species.
        waveform: Waveform to apply, or None for no DC voltages.
        duration: Duration of the simulation (s).
        dt: Timestep (s).
        initial_position: Initial position of the ion (µm).
        initial_velocity: Initial velocity of the ion (µm/s).
        full_rf: Specifies whether to simulate the full RF drive rather than the pseudopotential.

    Returns:
        Trajectory of the ion
    """
    trap._require_solved()
    n_it = int(np.ceil(duration / dt))
    pos = _array(initial_position)
    vel = _array(initial_velocity)
    acc = np.zeros(3)
    positions = np.zeros((n_it + 1, 3))
    velocities = np.zeros((n_it + 1, 3))
    _lib.velocity_verlet(
        C.byref(trap.struct), ion.mass, ion.charge, waveform._pointer if waveform is not None else None, int(full_rf),
        _ptr(pos), _ptr(vel), _ptr(acc), n_it, dt, _ptr(positions), _ptr(velocities)
    )
    flush_c_output()
    t_0 = waveform.times[0] if waveform is not None and waveform.n_steps > 0 else 0.0
    return Trajectory(t_0 + dt * np.arange(n_it + 1), positions, velocities)


def simulate_ensemble(trap, ion, waveform, positions, velocities, n_steps, dt, full_rf=False):
    """Integrate an ensemble of independent ions (e.g. sampled from a thermal state) in parallel.

    The ensemble runs on the CPU or on the OpenCL device (see set_device), starting from the start of the waveform.

    Args:
        trap: Trap whose electrodynamics have been solved.
        ion: Ion species.
        waveform: Waveform to apply, or None for no DC voltages.
        positions: Initial positions of the ions (µm) with shape (n_ions, 3).
        velocities: Initial velocities of the ions (µm/s) with shape (n_ions, 3).
        n_steps: Number of timesteps.
        dt: Timestep (s).
        full_rf: Specifies whether to simulate the full RF drive rather than the pseudopotential.

    Returns:
        Final positions (µm) and velocities (µm/s) of the ions with shape (n_ions, 3)
    """
    trap._require_solved()
    positions = _array(positions).reshape(-1, 3).copy()
    velocities = _array(velocities).reshape(-1, 3).copy()
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape")
    _lib.simulate_ensemble(
        C.byref(trap.struct), ion.mass, ion.charge, waveform._pointer if waveform is not None else None, int(full_rf),
        len(positions), _ptr(positions), _ptr(velocities), int(n_steps), dt
    )
    flush_c_output()
    return positions, velocities


def motional_quanta(ion, omega, position, velocity, equilibrium, axis=(1.0, 0.0, 0.0)):
    """Compute the number of quanta in the classical oscillation of an ion along a mode."""
    return _lib.motional_quanta(ion.mass, omega, _ptr(_array(axis)), _ptr(_array(position)), _ptr(_array(velocity)), _ptr(_array(equilibrium)))


def ideal_secular_frequencies(ion, V_DC, V_RF, Omega, Z_0):
    """Compute the secular frequencies (rad/s) of an ideal linear Paul trap with characteristic distance Z_0 (µm)."""
    omega = np.zeros(3)
    _lib.compute_secular_frequencies(ion.charge, ion.mass, V_DC, V_RF, Omega, Z_0, _ptr(omega))
    return omega


def heating_rate_rf_amplitude_noise(ion, Omega_RF, omega, dE2_dz, S_V_N, V_0):
    """Compute the heating rate (quanta/s) from RF amplitude noise for a squared field gradient dE2_dz (V^2/m^3)."""
    return _lib.heating_rate(ion.charge, ion.mass, Omega_RF, omega, dE2_dz, S_V_N, V_0)


def heating_rate_field_noise(ion, omega, S_E):
    """Compute the heating rate (quanta/s) from electric field noise S_E (V^2/m^2/Hz)."""
    return _lib.heating_rate_field_noise(ion.mass, ion.charge, omega, S_E)


def field_noise_psd(samples, dt, use_hann=True, subtract_mean=True):
    """Compute the one-sided power spectral density of field samples with ion_heating.c, where len(samples) must be a
    power of 2.
    """
    samples = _array(samples)
    options = SpectrumOptions(dt, len(samples), int(use_hann), int(subtract_mean))
    f_hz = _double_p()
    psd = _double_p()
    n_bins = _lib.ion_compute_psd(_ptr(samples), C.byref(options), C.byref(f_hz), C.byref(psd))
    if n_bins < 0:
        raise ValueError("the number of samples must be a power of 2 and at least 4")
    frequencies = _from_pointer(f_hz, n_bins)
    spectrum = _from_pointer(psd, n_bins)
    _libc.free(C.cast(f_hz, C.c_void_p))
    _libc.free(C.cast(psd, C.c_void_p))
    return frequencies, spectrum


# ---------------------------------------------------------------------------------------------------------------------
# Integrated photonics
# ---------------------------------------------------------------------------------------------------------------------

_POLARIZATIONS = {"TE": TE, "TM": TM}


def slab_mode(n_core, n_cover, n_substrate, thickness, wavelength, polarization="TE", order=0):
    """Solve for the effective index of a slab waveguide mode (lengths in µm), or None below cutoff."""
    n_eff = C.c_double()
    if _lib.solve_slab_mode(n_core, n_cover, n_substrate, thickness, wavelength, _POLARIZATIONS[polarization], order, C.byref(n_eff)) != 0:
        return None
    return n_eff.value


@dataclass
class Waveguide:
    """Rectangular channel waveguide (lengths in µm)."""

    wavelength: float = 0.729
    width: float = 0.6
    thickness: float = 0.15
    n_core: float = 2.0
    n_cladding: float = 1.455

    def _struct(self):
        return WaveguideStruct(self.wavelength, self.width, self.thickness, self.n_core, self.n_cladding, 0.0)

    def effective_index(self):
        """Solve for the effective index of the fundamental quasi-TE mode with the effective index method, or None if it
        is not guided.
        """
        struct = self._struct()
        if _lib.solve_waveguide_mode(C.byref(struct)) != 0:
            return None
        return struct.n_eff

    def dispersion(self, wavelength_min, wavelength_max, n_samples=21, degree=4):
        """Fit a polynomial to n_eff in u = (wavelength - center) / (half range), and compute the group index at the
        center.
        """
        struct = self._struct()
        coefficients = np.zeros(degree + 1)
        n_group = C.c_double()
        if _lib.fit_waveguide_dispersion(C.byref(struct), wavelength_min, wavelength_max, n_samples, degree, _ptr(coefficients), C.byref(n_group)) != 0:
            raise TiOptError("failed to fit waveguide dispersion")
        return coefficients, n_group.value


class Grating:
    """Grating coupler, which may share memory with a Trap."""

    FIELDS = ("azimuth", "period", "fill_factor", "etch_depth", "length", "width", "focal_distance", "power", "efficiency")

    def __init__(self, struct=None, **kwargs):
        if struct is None:
            struct = GratingCouplerStruct()
            struct.waveguide = Waveguide()._struct()
            struct.fill_factor = 0.5
            struct.etch_depth = struct.waveguide.thickness
            struct.length = 20.0
            struct.width = 8.0
            struct.power = 1e-3
            struct.efficiency = 0.3
        self.struct = struct
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __getattr__(self, name):
        if name in Grating.FIELDS:
            return getattr(self.struct, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in Grating.FIELDS:
            setattr(self.struct, name, float(value))
        elif name == "position":
            self.struct.position = _vec3(*value)
        elif name == "waveguide":
            self.struct.waveguide = value._struct()
        else:
            object.__setattr__(self, name, value)

    @property
    def position(self):
        """Point at which the beam leaves the trap surface (µm)."""
        return np.array(list(self.struct.position))

    @property
    def waveguide(self):
        """Waveguide which feeds the grating."""
        w = self.struct.waveguide
        return Waveguide(w.wavelength, w.width, w.thickness, w.n_core, w.n_cladding)

    def effective_index(self):
        """Compute the effective index of the grating region, averaging the etched and unetched sections by the fill
        factor, or None if they are not guided.
        """
        n_eff = C.c_double()
        if _lib.compute_grating_effective_index(C.byref(self.struct), C.byref(n_eff)) != 0:
            return None
        return n_eff.value

    def design(self, target):
        """Choose the azimuth, period, and focal distance so that the grating focuses on target (µm)."""
        if _lib.design_grating_coupler(C.byref(self.struct), _ptr(_array(target))) != 0:
            raise TiOptError("failed to design grating coupler")

    def emission(self):
        """Compute the emission angle from the surface normal (rad) and the unit beam direction, or None if no order
        radiates.
        """
        theta = C.c_double()
        k_hat = np.zeros(3)
        if _lib.compute_grating_emission(C.byref(self.struct), C.byref(theta), _ptr(k_hat)) != 0:
            return None
        return theta.value, k_hat

    def intensity(self, positions):
        """Evaluate the beam intensity (W/m^2) at positions with shape (..., 3)."""
        positions = _array(positions)
        shape = positions.shape[:-1]
        flat = _array(positions.reshape(-1, 3))
        intensity = np.zeros(len(flat))
        _lib.evaluate_grating_beam_batch(C.byref(self.struct), len(flat), _ptr(flat), _ptr(intensity))
        return intensity.reshape(shape)

    def beam_at(self, position):
        """Evaluate the beam intensity (W/m^2) and wavevector (1/m) at one position."""
        intensity = C.c_double()
        k_vector = np.zeros(3)
        if _lib.evaluate_grating_beam(C.byref(self.struct), _ptr(_array(position)), C.byref(intensity), _ptr(k_vector)) != 0:
            return 0.0, k_vector
        return intensity.value, k_vector


class Detector:
    """Integrated single-photon detector in the trap surface, which may share memory with a Trap."""

    FIELDS = ("width", "length", "wavelength", "quantum_efficiency", "dark_count_rate")

    def __init__(self, struct=None, **kwargs):
        if struct is None:
            struct = DetectorStruct()
            struct.width = 50.0
            struct.length = 50.0
            struct.wavelength = 0.397
            struct.quantum_efficiency = 0.8
            struct.dark_count_rate = 10.0
        self.struct = struct
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __getattr__(self, name):
        if name in Detector.FIELDS:
            return getattr(self.struct, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in Detector.FIELDS:
            setattr(self.struct, name, float(value))
        elif name == "position":
            self.struct.position = _vec3(*value)
        else:
            object.__setattr__(self, name, value)

    @property
    def position(self):
        """Center of the active area (µm)."""
        return np.array(list(self.struct.position))

    def collection_efficiency(self, position):
        """Compute the fraction of the photons emitted by an ion at position (µm) which the detector counts."""
        return _lib.compute_collection_efficiency(C.byref(self.struct), _ptr(_array(position)))


def scattering_rate(gamma, s, detuning=0.0):
    """Compute the photon scattering rate (1/s) of a transition with linewidth gamma (1/s), saturation parameter s, and
    detuning (rad/s).
    """
    return _lib.compute_scattering_rate(gamma, s, detuning)


def detection_fidelity(photon_rate, dark_count_rate, decay_rate, detection_time):
    """Compute the fidelity of threshold detection and the best threshold.

    Args:
        photon_rate: Count rate of the bright state (1/s).
        dark_count_rate: Rate of dark counts (1/s).
        decay_rate: Rate at which the dark state decays into the bright state during detection (1/s).
        detection_time: Duration of detection (s).

    Returns:
        Fidelity and best threshold
    """
    threshold = C.c_int()
    fidelity = _lib.compute_detection_fidelity(photon_rate, dark_count_rate, decay_rate, detection_time, C.byref(threshold))
    return fidelity, threshold.value


def gaussian_beam(waist_aperture, focal_distance, wavelength):
    """Compute the waist of a Gaussian beam focused from an aperture, and its distance from the aperture."""
    waist = C.c_double()
    distance = C.c_double()
    _lib.compute_gaussian_beam(waist_aperture, focal_distance, wavelength, C.byref(waist), C.byref(distance))
    return waist.value, distance.value


def multilayer_response(indices, thicknesses, wavelength, theta=0.0, polarization="TE"):
    """Compute the complex reflection and transmission coefficients and the reflectance and transmittance of a stack of
    layers, where indices may be complex and the first and last layers are semi-infinite.
    """
    indices = np.asarray(indices, dtype=complex)
    n_real = _array(indices.real)
    n_imag = _array(indices.imag)
    d = _array(thicknesses)
    r = np.zeros(2)
    t = np.zeros(2)
    R = C.c_double()
    T = C.c_double()
    if _lib.compute_multilayer_response(len(indices), _ptr(n_real), _ptr(n_imag), _ptr(d), wavelength, theta, _POLARIZATIONS[polarization], _ptr(r), _ptr(t), C.byref(R), C.byref(T)) != 0:
        raise ValueError("a stack needs at least two layers")
    return complex(*r), complex(*t), R.value, T.value


def grating_directionality(n_box, box_thickness, n_substrate, wavelength, theta_box, polarization="TE"):
    """Compute the fraction of the power radiated by a grating which leaves upwards, given a reflecting substrate."""
    n_substrate = complex(n_substrate)
    return _lib.compute_grating_directionality(n_box, box_thickness, n_substrate.real, n_substrate.imag, wavelength, theta_box, _POLARIZATIONS[polarization])


def fit_polynomial(x, y, degree):
    """Fit polynomial coefficients (lowest order first) by least squares, and return them with chi squared."""
    x = _array(x)
    y = _array(y)
    coefficients = np.zeros(degree + 1)
    chi_sq = C.c_double()
    if _lib.fit_polynomial(len(x), _ptr(x), _ptr(y), degree, _ptr(coefficients), C.byref(chi_sq)) != 0:
        raise TiOptError("polynomial fit failed")
    return coefficients, chi_sq.value


def fit_gaussian(x, y):
    """Fit the (amplitude, center, sigma, offset) of a Gaussian by least squares."""
    x = _array(x)
    y = _array(y)
    parameters = np.zeros(4)
    _lib.fit_gaussian(len(x), _ptr(x), _ptr(y), _ptr(parameters))
    return parameters


# ---------------------------------------------------------------------------------------------------------------------
# Quantum dynamics
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Pulse:
    """Laser pulse on the S-D transition.

    The pulse has a start time (s), duration (s), detuning from the transition (rad/s), Rabi frequency (rad/s), phase
    (rad), target ions (starting at 1), and the Lamb-Dicke parameter of each target.
    """

    start: float
    duration: float
    detuning: float
    rabi_frequency: float
    phase: float = 0.0
    targets: list = field(default_factory=lambda: [1])
    lamb_dicke: float = 0.1


@dataclass
class QuantumResult:
    """Result of simulate_pulses, i.e. the sample times (s), the excited population of each ion, the mean phonon number,
    and optionally the final density matrix.
    """
    times: np.ndarray
    excited_populations: np.ndarray
    mean_phonon_numbers: np.ndarray
    density_matrix: np.ndarray = None


def simulate_pulses(ion, pulses, n_ions=1, mode_frequency=AXIAL_FREQUENCY, n_max=15, n_bar=0.0, alpha=0.0,
                    heating_rate=0.0, dephasing_rate=0.0, decay=True, n_samples=200, density_matrix=False):
    """Simulate laser pulses on ions sharing one motional mode with the master equation (see simulate_pulse_program).

    Args:
        ion: Ion species.
        pulses: List of Pulse objects.
        n_ions: Number of ions.
        mode_frequency: Angular frequency of the motional mode (rad/s).
        n_max: Highest Fock state kept in the motional mode.
        n_bar: Mean phonon number of the initial thermal state of the mode.
        alpha: Coherent displacement of the initial state of the mode.
        heating_rate: Heating rate of the mode (quanta/s).
        dephasing_rate: Dephasing rate of the qubits (1/s).
        decay: Specifies whether to include spontaneous decay of the qubits.
        n_samples: Number of times at which the populations are sampled.
        density_matrix: Specifies whether to return the final density matrix.

    Returns:
        Result of the simulation
    """
    program = _lib.allocate_pulse_program(n_ions, len(pulses))
    targets_memory = []
    try:
        p_struct = program.contents
        rabi = np.zeros(len(pulses))
        etas = np.zeros((len(pulses), n_ions))
        for p, pulse in enumerate(pulses):
            p_struct.pulse_frequencies[p] = ion.qubit_frequency + pulse.detuning
            p_struct.pulse_times[p] = pulse.start
            p_struct.pulse_durations[p] = pulse.duration
            p_struct.pulse_phases[p] = pulse.phase
            # The targets are freed by free_pulse_program, so they must be allocated by C
            targets = C.cast(_libc.malloc(C.sizeof(C.c_int) * max(len(pulse.targets), 1)), _int_p)
            for t, target in enumerate(pulse.targets):
                if not 1 <= target <= n_ions:
                    raise ValueError(f"pulse {p} targets ion {target}, but there are only {n_ions} ions")
                targets[t] = int(target)
            p_struct.pulse_targets_list[p] = targets
            p_struct.pulse_n_targets_list[p] = len(pulse.targets)
            rabi[p] = pulse.rabi_frequency
            etas[p, :] = pulse.lamb_dicke

        alpha = complex(alpha)
        mode = MotionalModeStruct(mode_frequency, n_max, n_bar, (C.c_double * 2)(alpha.real, alpha.imag), heating_rate)

        ion_struct = IonParameters.from_buffer_copy(ion.struct)
        if not decay:
            ion_struct.gamma_SD = 0.0

        times = np.zeros(n_samples)
        excited = np.zeros((n_samples, n_ions))
        n_mean = np.zeros(n_samples)
        dim = (2 ** n_ions) * (n_max + 1)
        rho = np.zeros((dim * dim, 2)) if density_matrix else None

        status = _lib.simulate_pulse_program(
            C.byref(ion_struct), C.byref(mode), program, _ptr(rabi), _ptr(_array(etas)), dephasing_rate,
            n_samples, _ptr(times), _ptr(excited), _ptr(n_mean), _ptr(rho) if density_matrix else None
        )
        flush_c_output()
        if status != 0:
            raise TiOptError("quantum simulation failed")
    finally:
        _lib.free_pulse_program(program)

    result = QuantumResult(times, excited, n_mean)
    if density_matrix:
        result.density_matrix = (rho[:, 0] + 1j * rho[:, 1]).reshape(dim, dim)
    return result


def thermal_rabi_excitation(rabi_frequency, eta, sideband, n_bar, detuning, times, n_max=50):
    """Compute the analytical excitation of a thermal ion driven on a sideband (where 0 is the carrier)."""
    return np.array([_lib.thermal_rabi_excitation(rabi_frequency, eta, sideband, n_bar, detuning, t, n_max) for t in np.atleast_1d(times)])


def displacement_matrix_element(m, n, alpha):
    """Compute the matrix element <m|D(alpha)|n> of the displacement operator."""
    element = np.zeros(2)
    alpha = complex(alpha)
    _lib.displacement_matrix_element(m, n, alpha.real, alpha.imag, _ptr(element))
    return complex(*element)


def lamb_dicke_parameter(k_vector, mode_axis, ion, omega):
    """Compute the Lamb-Dicke parameter of a beam with wavevector k_vector (1/m) and a mode along mode_axis with angular
    frequency omega (rad/s).
    """
    return _lib.lamb_dicke_parameter(_ptr(_array(k_vector)), _ptr(_array(mode_axis)), ion.mass, omega)


def saturation_intensity(gamma, wavelength):
    """Compute the saturation intensity (W/m^2) of a transition with linewidth gamma (1/s) and wavelength (m)."""
    return _lib.saturation_intensity(gamma, wavelength)


def rabi_frequency_from_intensity(gamma, wavelength, intensity):
    """Compute the Rabi frequency (rad/s) of a transition with linewidth gamma (1/s) and wavelength (m) at an intensity
    (W/m^2).
    """
    return _lib.rabi_frequency_from_intensity(gamma, wavelength, intensity)


# ---------------------------------------------------------------------------------------------------------------------
# Circuits and transport scheduling
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Instruction:
    """Instruction of a circuit, with its operation, target ions, layer, and the instructions which come before it."""
    operation: str
    targets: list
    layer: int
    previous: list


class Circuit:
    """Quantum circuit parsed from OpenQASM, with its instructions assigned to layers."""

    def __init__(self, path):
        self._pointer = _lib.parse_qasm(str(path).encode())
        flush_c_output()
        if not self._pointer:
            raise TiOptError(f"failed to parse {path}")

    def close(self):
        """Free the memory of the circuit."""
        if self._pointer:
            _lib.free_circuit_graph(self._pointer)
            self._pointer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def n_ions(self):
        """Number of ions (qubits) in the circuit."""
        return self._pointer.contents.n_ions

    @property
    def instructions(self):
        """Instructions of the circuit."""
        graph = self._pointer.contents
        instructions = []
        for i in range(graph.n_instruction_nodes):
            node = graph.instruction_nodes[i]
            instructions.append(Instruction(
                OPERATION_TYPES.get(node.operation_type, str(node.operation_type)),
                [node.target_ions[t] for t in range(node.n_target_ions)],
                graph.layer_idxs[i],
                [node.prev_nodes[p].graphindex for p in range(node.n_prev_nodes)],
            ))
        return instructions


def _zone_type(name):
    value = 0
    for part in name.upper().split("+"):
        if part.strip() not in ZONE_TYPES:
            raise ValueError(f"unknown zone type {part.strip().lower()!r}; choose from {', '.join(k.lower() for k in ZONE_TYPES)}")
        value |= ZONE_TYPES[part.strip()]
    return value


class TrapGraph:
    """Zones along a linear trap, between which ions are transported."""

    def __init__(self, zone_types, zone_capacities, x_min, x_max, height=ION_HEIGHT):
        """Make zones spaced evenly along the trap axis.

        Args:
            zone_types: Type of each zone, where a zone can have several types joined by "+" (e.g. "load+storage").
            zone_capacities: Number of ions which each zone can hold.
            x_min: Position of the first zone along the trap axis (µm).
            x_max: Position of the last zone along the trap axis (µm).
            height: Height of the zones above the surface (µm).
        """
        n = len(zone_types)
        types = (C.c_int * n)(*[_zone_type(t) for t in zone_types])
        capacities = (C.c_int * n)(*[int(c) for c in zone_capacities])
        self.struct = TrapGraphStruct()
        _lib.generate_linear_trap_graph(n, types, capacities, x_min, x_max, height, C.byref(self.struct))
        self.zone_types = list(zone_types)

    def close(self):
        """Free the memory of the zones."""
        if self.struct.zone_nodes:
            _lib.free_trap_graph(C.byref(self.struct))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def positions(self):
        """Positions of the zones (µm)."""
        return np.array([list(self.struct.zone_nodes[z].position) for z in range(self.struct.n_zones)])


class TransportProgram:
    """Transports which bring the ions of a circuit to the zones where their operations happen."""

    def __init__(self, trap_graph, circuit):
        """Schedule the transports of a circuit on the zones of a TrapGraph."""
        memory = _libc.malloc(C.sizeof(TransportProgramStruct))
        C.memset(memory, 0, C.sizeof(TransportProgramStruct))
        pointer = _lib.generate_transport_schedule(C.byref(trap_graph.struct), circuit._pointer, C.cast(memory, C.POINTER(TransportProgramStruct)))
        flush_c_output()
        if not pointer:
            _libc.free(memory)
            raise TiOptError("failed to schedule transports (check zone types and capacities)")
        self._pointer = pointer

    def close(self):
        """Free the memory of the transport program."""
        if self._pointer:
            _lib.free_transport_program(self._pointer)
            self._pointer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def n_ions(self):
        """Number of ions."""
        return self._pointer.contents.n_ions

    @property
    def n_transports(self):
        """Number of transports."""
        return self._pointer.contents.n_transports

    @property
    def arrival_times(self):
        """Time at which each transport arrives (s)."""
        return _from_pointer(self._pointer.contents.arrival_times, self.n_transports)

    @property
    def destinations(self):
        """Destination of each transport (µm)."""
        return _from_pointer(C.cast(self._pointer.contents.destinations, _double_p), 3 * self.n_transports).reshape(-1, 3)

    @property
    def initial_positions(self):
        """Position of each ion before the first transport (µm)."""
        return _from_pointer(C.cast(self._pointer.contents.initial_positions, _double_p), 3 * self.n_ions).reshape(-1, 3)

    @property
    def targets(self):
        """Ions moved by each transport."""
        program = self._pointer.contents
        return [[program.transport_targets_list[t][i] for i in range(program.transport_n_targets_list[t])] for t in range(self.n_transports)]


# ---------------------------------------------------------------------------------------------------------------------
# Acceleration
# ---------------------------------------------------------------------------------------------------------------------


def set_num_threads(n_threads=0):
    """Set the number of threads used by the library (0 restores one per core)."""
    _lib.set_num_threads(int(n_threads))


def get_num_threads():
    """Get the number of threads used by the library."""
    return _lib.get_num_threads()


def opencl_available():
    """Check whether an OpenCL device with double precision can be used, which is a GPU (or any device if
    $TIOPT_OPENCL_ANY_DEVICE is set), where $TIOPT_OPENCL_DEVICE picks among several.
    """
    available = bool(_lib.opencl_available())
    flush_c_output()
    return available


def opencl_device_name():
    """Get the name of the OpenCL device, or an empty string if there is none."""
    name = _lib.opencl_device_name()
    return name.decode() if name else ""


def set_device(device):
    """Run relaxation, potential maps, and ensembles on "cpu" or "opencl"."""
    devices = {"cpu": CPU_DEVICE, "opencl": OPENCL_DEVICE}
    if device not in devices:
        raise ValueError(f"unknown device {device!r}; choose from {tuple(devices)}")
    status = _lib.set_compute_device(devices[device])
    flush_c_output()
    if status != 0:
        raise TiOptError("no OpenCL device with double precision is available")


def get_device():
    """Get the device which relaxation, potential maps, and ensembles run on ("cpu" or "opencl")."""
    return "opencl" if _lib.get_compute_device() == OPENCL_DEVICE else "cpu"


# ---------------------------------------------------------------------------------------------------------------------
# Co-design
# ---------------------------------------------------------------------------------------------------------------------


def _struct_to_dict(struct):
    return {name: getattr(struct, name) for name, _ in struct._fields_}


def _dict_to_struct(struct_type, values, base):
    struct = struct_type()
    C.memmove(C.byref(struct), C.byref(base), C.sizeof(struct_type))
    for name, value in values.items():
        if name not in dict(struct_type._fields_):
            raise KeyError(f"unknown field {name!r} of {struct_type.__name__}")
        setattr(struct, name, value)
    return struct


def default_codesign_parameters():
    """Get the default parameters of the co-design flow (see struct CodesignParameters)."""
    struct = CodesignParametersStruct()
    _lib.default_codesign_parameters(C.byref(struct))
    return _struct_to_dict(struct)


def default_codesign_environment():
    """Get the default environment of the co-design flow (see struct CodesignEnvironment)."""
    struct = CodesignEnvironmentStruct()
    _lib.default_codesign_environment(C.byref(struct))
    return _struct_to_dict(struct)


@dataclass
class CodesignRecord:
    """Design after one stage of one pass of the co-design flow."""

    pass_index: int
    stage: str
    n_evaluations: int
    objective: float
    parameters: dict
    metrics: dict


class Codesign:
    """Co-design of a trap, its photonics, transport program, and waveforms for the circuit in a QASM file.

    Stage 0 makes a basic design of everything when the co-design is created, and run() then makes passes over stages
    1-5 (trap, photonics, transport, waveforms, and metrics). The first pass optimizes each part a little based on its
    own objective, and later passes optimize every part based on the quantum computing metrics of the whole design.
    """

    def __init__(self, qasm_path, element="40Ca+", environment=None, parameters=None):
        """Make the basic design for a circuit (stage 0).

        Args:
            qasm_path: OpenQASM file of the circuit.
            element: Ion species.
            environment: Values which replace the defaults of default_codesign_environment.
            parameters: Values which replace the defaults of default_codesign_parameters.
        """
        env = _dict_to_struct(CodesignEnvironmentStruct, environment or {}, _environment_defaults())
        params = _dict_to_struct(CodesignParametersStruct, parameters or {}, _parameter_defaults())
        pointer = _lib.create_codesign(element.encode(), str(qasm_path).encode(), C.byref(env), C.byref(params))
        flush_c_output()
        if not pointer:
            raise TiOptError(f"failed to set up the co-design of {qasm_path}")
        self._pointer = pointer
        self._trap = None

    def close(self):
        """Free the memory of the co-design."""
        if self._pointer:
            _lib.free_codesign(self._pointer)
            self._pointer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _state(self):
        parameters = CodesignParametersStruct()
        environment = CodesignEnvironmentStruct()
        metrics = QCMetricsStruct()
        _lib.get_codesign_state(self._pointer, C.byref(parameters), C.byref(environment), C.byref(metrics))
        return parameters, environment, metrics

    @property
    def parameters(self):
        """Current parameters of the design (see struct CodesignParameters), which can also be set."""
        return _struct_to_dict(self._state()[0])

    @parameters.setter
    def parameters(self, values):
        struct = _dict_to_struct(CodesignParametersStruct, values, self._state()[0])
        _lib.set_codesign_state(self._pointer, C.byref(struct), None)

    @property
    def environment(self):
        """Hardware and noise which the design assumes (see struct CodesignEnvironment), which can also be set."""
        return _struct_to_dict(self._state()[1])

    @environment.setter
    def environment(self, values):
        struct = _dict_to_struct(CodesignEnvironmentStruct, values, self._state()[1])
        _lib.set_codesign_state(self._pointer, None, C.byref(struct))

    @property
    def metrics(self):
        """Metrics of the design from its last evaluation (see struct QCMetrics)."""
        return _struct_to_dict(self._state()[2])

    def evaluate(self, simulate_gates=True):
        """Evaluate the metrics of the current design, optionally simulating the two-qubit gates with the master
        equation.
        """
        metrics = QCMetricsStruct()
        _lib.evaluate_codesign(self._pointer, int(simulate_gates), C.byref(metrics))
        flush_c_output()
        return _struct_to_dict(metrics)

    def optimize_stage(self, stage, qc_aware=True, max_evaluations=10):
        """Optimize one stage of the design a little.

        Args:
            stage: Stage to optimize, which is "trap", "photonics", "transport", or "waveforms".
            qc_aware: Specifies whether to optimize the metrics of the whole circuit rather than the figures of merit of
                the stage.
            max_evaluations: Largest number of evaluations of the design.

        Returns:
            Objective value which was reached and the number of evaluations made
        """
        objective = C.c_double()
        n_evaluations = C.c_int()
        _lib.optimize_codesign_stage(self._pointer, CODESIGN_STAGES.index(stage), int(qc_aware), int(max_evaluations), C.byref(objective), C.byref(n_evaluations))
        flush_c_output()
        return objective.value, n_evaluations.value

    def run(self, n_passes=2, max_evaluations=10):
        """Run the co-design flow, whose history can be read from another thread while it runs.

        Args:
            n_passes: Number of passes, where the first pass optimizes each stage based on its own objective and every
                later pass is QC-aware.
            max_evaluations: Largest number of evaluations of the design in each stage.
        """
        _lib.run_codesign(self._pointer, int(n_passes), int(max_evaluations))
        flush_c_output()

    @property
    def n_records(self):
        """Number of records in the history of the flow."""
        return _lib.get_codesign_n_records(self._pointer)

    def records(self, start=0):
        """Get the history of the flow as CodesignRecord objects, starting from record start."""
        records = []
        for r in range(start, self.n_records):
            record = CodesignRecordStruct()
            _lib.get_codesign_record(self._pointer, r, C.byref(record))
            records.append(CodesignRecord(
                record.pass_, CODESIGN_STAGES[record.stage], record.n_evaluations, record.objective,
                _struct_to_dict(record.parameters), _struct_to_dict(record.metrics),
            ))
        return records

    def layout(self):
        """Get the positions of the gate and measurement zones (µm) and the numbers of zones, layers, and ions."""
        gate = np.zeros(3)
        measure = np.zeros(3)
        n_zones, n_layers, n_ions = C.c_int(), C.c_int(), C.c_int()
        _lib.get_codesign_layout(self._pointer, _ptr(gate), _ptr(measure), C.byref(n_zones), C.byref(n_layers), C.byref(n_ions))
        return {"gate_position": gate, "measure_position": measure, "n_zones": n_zones.value, "n_layers": n_layers.value, "n_ions": n_ions.value}

    def trap(self):
        """Get the current trap, which shares memory with the co-design (it is replaced whenever the trap stage changes
        its geometry, so it shouldn't be kept across calls).
        """
        trap = Trap.__new__(Trap)
        trap.struct = _lib.get_codesign_trap(self._pointer).contents
        trap.solved = True
        trap.solver = "analytical"
        trap._owner = self
        trap.close = lambda: None
        return trap

    def save_trap(self, path):
        """Write the current trap to a trap file."""
        if _lib.write_trap_to_file(str(path).encode(), _lib.get_codesign_trap(self._pointer)) != 0:
            raise TiOptError(f"failed to write trap file {path}")


def _environment_defaults():
    struct = CodesignEnvironmentStruct()
    _lib.default_codesign_environment(C.byref(struct))
    return struct


def _parameter_defaults():
    struct = CodesignParametersStruct()
    _lib.default_codesign_parameters(C.byref(struct))
    return struct
