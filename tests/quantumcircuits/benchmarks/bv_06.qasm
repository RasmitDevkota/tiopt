// Bernstein-Vazirani with secret 11001 and an ancilla, on 6 qubits
qubit[6] qreg;
bit[6] creg;

x qreg[5];
h qreg[0];
h qreg[1];
h qreg[2];
h qreg[3];
h qreg[4];
h qreg[5];
cx qreg[0], qreg[5];
cx qreg[1], qreg[5];
cx qreg[4], qreg[5];
h qreg[0];
h qreg[1];
h qreg[2];
h qreg[3];
h qreg[4];

measure qreg -> creg;
