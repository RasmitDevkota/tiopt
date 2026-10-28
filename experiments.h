#pragma once

#include "data_structures.h"

enum Experiment
{
	IDLE,
	BACK_AND_FORTH,
};

void run_experiment
(
	struct Trap *trap,
	struct IonParameters *ion,
	enum Experiment experiment,
	double *cost
);
