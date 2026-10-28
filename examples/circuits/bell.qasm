// Bell pair: a Hadamard and a CNOT entangle two qubits, which are then measured
qubit[2] qreg;
bit[2] creg;

h qreg[0];
cx qreg[0], qreg[1];

measure qreg -> creg;
