// GHZ state on 12 qubits
qubit[12] qreg;
bit[12] creg;

h qreg[0];
cx qreg[0], qreg[1];
cx qreg[1], qreg[2];
cx qreg[2], qreg[3];
cx qreg[3], qreg[4];
cx qreg[4], qreg[5];
cx qreg[5], qreg[6];
cx qreg[6], qreg[7];
cx qreg[7], qreg[8];
cx qreg[8], qreg[9];
cx qreg[9], qreg[10];
cx qreg[10], qreg[11];

measure qreg -> creg;
