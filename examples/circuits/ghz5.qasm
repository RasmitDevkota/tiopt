// Five-qubit GHZ state, which needs more shuttling between the storage and gate zones
qubit[5] qreg;
bit[5] creg;

h qreg[0];
cx qreg[0], qreg[1];
cx qreg[1], qreg[2];
cx qreg[2], qreg[3];
cx qreg[3], qreg[4];

measure qreg -> creg;
