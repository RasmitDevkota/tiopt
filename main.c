#include "main.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <nlopt.h>

#include "defs.h"
#include "io.h"
#include "data_structures.h"
#include "electrodynamics.h"
#include "verlet.h"

// Cost function passed to optimization routine
double cost_function(
    unsigned n,
    const double *x,
    double *grad,
    void *trap_obj
) {
    // The purpose of this function is to compute an aggregate cost of a trap design and transport configuration
    double cost = 0.0;

    // Being lazy with memory for now just as a sketch
    struct Trap *trap = (struct Trap *)trap_obj;

    // %TODO - update trap based on optimization parameters

    // Solve for the potential energy as a 3D array of data for each electrode individually
    solve_trap_electrodynamics(trap, RELAXATION);

    // Perform ion transport experiments now! we don't know how we will actually structure this part, just an example
    for (int e = 0; e < 5; e++)
    {
        // cost_experiment_e = perform_experiment(trap, e);
        // cost += cost_experiment_e;
    }

    cost += 0.5;

    return cost;
}

// Main optimization loop
int main()
{
    printf(
        "|=---------------------------------------------------------------------------------------------------|\n"
        "|                                                                                                    |\n"
        "|                                   ████████╗██╗ ██████╗ ██████╗ ████████╗                           |\n"
        "|                                   ╚══██╔══╝██║██╔═══██╗██╔══██╗╚══██╔══╝                           |\n"
        "|                                      ██║   ██║██║   ██║██████╔╝   ██║                              |\n"
        "|                                      ██║   ██║██║   ██║██╔═══╝    ██║                              |\n"
        "|                                      ██║   ██║╚██████╔╝██║        ██║                              |\n"
        "|                                      ╚═╝   ╚═╝ ╚═════╝ ╚═╝        ╚═╝                              |\n"
        "|                                                                                                    |\n"
        "|                                                                                                    |\n"
        "|                                                                                                    |\n"
        "|                                                 XXXXXXX                                            |\n"
        "|                                               XX       XX                                          |\n"
        "|                                              X           X             \\                           |\n"
        "|                                             XX           XX   ==========》                         |\n"
        "|                                              X           X             /                           |\n"
        "|                                               XX       XX                                          |\n"
        "|                                                 XXXXXXX                                            |\n"
        "|                                                                                                    |\n"
        "|                                                                                                    |\n"
        "|                                 ########################################################           |\n"
        "|                                ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++###           |\n"
        "|                              ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.##           |\n"
        "|                            ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##...##           |\n"
        "|                          ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.....##           |\n"
        "|                        #########################################################......##           |\n"
        "|                      #########################################################......##             |\n"
        "|                    #########################################################......##               |\n"
        "|                  #########################################################......##                 |\n"
        "|                ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.......##                   |\n"
        "|              ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.......##                     |\n"
        "|            ##+++++++##+++++++##+++++++##+++++++##+++++++##+++++++##.......##                       |\n"
        "|          ########################################################.......##                         |\n"
        "|          ##....................................................##.....##                           |\n"
        "|          ##....................................................##...##                             |\n"
        "|          ##....................................................##.##                               |\n"
        "|          #########################################################                                 |\n"
        "|                                                                                                    |\n"
        "|----------------------------------------------------------------------------------------------------|\n"
    );

	printf("SPH_R=%f\n", SPH_R);
	printf("SPH_Z_MIN=%f\n", SPH_Z_MIN);
	printf("SPH_SPACING=%f\n", SPH_SPACING);
	printf("NSPH_X=%d, NSPH_Y=%d\n", NSPH_X, NSPH_Y);
	printf("RELAXATION_NX=%d, RELAXATION_NY=%d, RELAXATION_NZ=%d\n", RELAXATION_NX, RELAXATION_NY, RELAXATION_NZ);

    double min_cost = 99999;
    const int dim = 12;

    printf("- Constructing optimization problem...\n");

    // nlopt setup
    nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);

    printf("Setting constraints...\n");
    // @TODO - constraints

    printf("Setting bounds...\n");
    // @TODO - bounds

    printf("Setting stopping criteria...\n");
    // @TODO - tolerances

	// @TEST - limit number of evaluations to 1 for testing purposes
	nlopt_set_maxeval(opt, 1);

    printf("Constructing initial guess...\n");
    // Generate initial guess trap from file
    char *trap_filename = "trap.data";
    struct Trap trap;
    generate_trap_from_file(trap_filename, &trap);

    double x[dim];
    for (int i = 0; i < dim; i++)
    {
        x[i] = 0.0;
    }

    printf("Setting objectives...\n");
    nlopt_set_min_objective(opt, cost_function, &trap);

    printf("Calling objective once...\n");
    cost_function(
      dim,
      x,
      NULL,
      &trap
    );

    printf("Optimizing...\n");
    if (nlopt_optimize(opt, x, &min_cost) < 0)
    {
        printf("nlopt failed!\n");
    }
    else
    {
        printf("Found minimum at f(");
        for (int k = 0; k < dim; k++)
        {
            printf("%g", x[k]);

            if (k < dim-1)
            {
                printf(",");
            }
        }
        printf(") = %0.10g\n", min_cost);
    }

    // Free malloc'd memory
    free(trap.electrode_positions);

    for (int e = 0; e < trap.n_electrodes; e++)
    {
        free((*trap.electrodes)[e].vertices);
        free((*trap.electrodes)[e].edges);
        free((*trap.electrodes)[e].Vlm);
    }

    free(trap.electrodes);

    return 0;
}

