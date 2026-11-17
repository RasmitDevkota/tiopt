qubit[3] qreg;
bit[3] creg;

cx qreg[0], qreg[1];
cx qreg[1], qreg[2];
cx qreg[0], qreg[1];

measure qreg -> creg;
