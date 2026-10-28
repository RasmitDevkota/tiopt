// Three-qubit GHZ state, built by a chain of CNOTs
qubit[3] qreg;
bit[3] creg;

h qreg[0];
cx qreg[0], qreg[1];
cx qreg[1], qreg[2];

measure qreg -> creg;
