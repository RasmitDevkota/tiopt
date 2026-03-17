qubit[4] qreg;
bit[4] creg;

h qreg[0];
x qreg[1];
y qreg[2];
z qreg[3];

cx qreg[0], qreg[1];
cx qreg[2], qreg[3];

cy qreg[0], qreg[2];
cz qreg[1], qreg[3];

cz qreg[0], qreg[3];
cz qreg[1], qreg[2];

measure qreg, creg;

