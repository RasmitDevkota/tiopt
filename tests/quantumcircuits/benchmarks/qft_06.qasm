// Quantum Fourier transform on 6 qubits
qubit[6] qreg;
bit[6] creg;

h qreg[0];
cp(1.570796) qreg[1], qreg[0];
cp(0.785398) qreg[2], qreg[0];
cp(0.392699) qreg[3], qreg[0];
cp(0.196350) qreg[4], qreg[0];
cp(0.098175) qreg[5], qreg[0];
h qreg[1];
cp(1.570796) qreg[2], qreg[1];
cp(0.785398) qreg[3], qreg[1];
cp(0.392699) qreg[4], qreg[1];
cp(0.196350) qreg[5], qreg[1];
h qreg[2];
cp(1.570796) qreg[3], qreg[2];
cp(0.785398) qreg[4], qreg[2];
cp(0.392699) qreg[5], qreg[2];
h qreg[3];
cp(1.570796) qreg[4], qreg[3];
cp(0.785398) qreg[5], qreg[3];
h qreg[4];
cp(1.570796) qreg[5], qreg[4];
h qreg[5];
swap qreg[0], qreg[5];
swap qreg[1], qreg[4];
swap qreg[2], qreg[3];

measure qreg -> creg;
