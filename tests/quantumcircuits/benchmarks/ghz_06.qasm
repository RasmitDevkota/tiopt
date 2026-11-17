// GHZ state on 6 qubits
qubit[6] qreg;
bit[6] creg;

h qreg[0];
cx qreg[0], qreg[1];
cx qreg[1], qreg[2];
cx qreg[2], qreg[3];
cx qreg[3], qreg[4];
cx qreg[4], qreg[5];

measure qreg -> creg;
