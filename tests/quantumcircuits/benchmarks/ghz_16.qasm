// GHZ state on 16 qubits
qubit[16] qreg;
bit[16] creg;

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
cx qreg[11], qreg[12];
cx qreg[12], qreg[13];
cx qreg[13], qreg[14];
cx qreg[14], qreg[15];

measure qreg -> creg;
