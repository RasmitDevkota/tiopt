// Trotterized Heisenberg chain of 6 spins, 2 steps
qubit[6] qreg;
bit[6] creg;

rxx(0.2) qreg[0], qreg[1];
ryy(0.2) qreg[0], qreg[1];
rzz(0.2) qreg[0], qreg[1];
rxx(0.2) qreg[2], qreg[3];
ryy(0.2) qreg[2], qreg[3];
rzz(0.2) qreg[2], qreg[3];
rxx(0.2) qreg[4], qreg[5];
ryy(0.2) qreg[4], qreg[5];
rzz(0.2) qreg[4], qreg[5];
rxx(0.2) qreg[1], qreg[2];
ryy(0.2) qreg[1], qreg[2];
rzz(0.2) qreg[1], qreg[2];
rxx(0.2) qreg[3], qreg[4];
ryy(0.2) qreg[3], qreg[4];
rzz(0.2) qreg[3], qreg[4];
rxx(0.2) qreg[0], qreg[1];
ryy(0.2) qreg[0], qreg[1];
rzz(0.2) qreg[0], qreg[1];
rxx(0.2) qreg[2], qreg[3];
ryy(0.2) qreg[2], qreg[3];
rzz(0.2) qreg[2], qreg[3];
rxx(0.2) qreg[4], qreg[5];
ryy(0.2) qreg[4], qreg[5];
rzz(0.2) qreg[4], qreg[5];
rxx(0.2) qreg[1], qreg[2];
ryy(0.2) qreg[1], qreg[2];
rzz(0.2) qreg[1], qreg[2];
rxx(0.2) qreg[3], qreg[4];
ryy(0.2) qreg[3], qreg[4];
rzz(0.2) qreg[3], qreg[4];

measure qreg -> creg;
