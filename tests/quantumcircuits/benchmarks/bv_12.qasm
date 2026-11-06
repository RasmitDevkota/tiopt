// Bernstein-Vazirani with secret 11000100000 and an ancilla, on 12 qubits
qubit[12] qreg;
bit[12] creg;

x qreg[11];
h qreg[0];
h qreg[1];
h qreg[2];
h qreg[3];
h qreg[4];
h qreg[5];
h qreg[6];
h qreg[7];
h qreg[8];
h qreg[9];
h qreg[10];
h qreg[11];
cx qreg[0], qreg[11];
cx qreg[1], qreg[11];
cx qreg[5], qreg[11];
h qreg[0];
h qreg[1];
h qreg[2];
h qreg[3];
h qreg[4];
h qreg[5];
h qreg[6];
h qreg[7];
h qreg[8];
h qreg[9];
h qreg[10];

measure qreg -> creg;
