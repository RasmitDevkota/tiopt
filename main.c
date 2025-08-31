#include "main.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <nlopt.h>

#include "defs.h"
#include "data_structures.h"
#include "io.h"
#include "trap_geometry.h"
#include "electrodynamics.h"
#include "verlet.h"

// cost function passed to optimization routine
double cost_function(
    unsigned n,
    const double *x,
    double *grad,
    void *trap_obj
) {
    // the purpose of this function is to compute an aggregate cost of a trap design and transport configuration
    double cost = 0.0;

    // being lazy with memory for now just as a sketch
    struct Trap *trap = (struct Trap *)trap_obj;

    // %TODO - update trap based on optimization parameters

    // generate the initial mesh object needed from the trap electrodes
    generate_mesh(trap);

    // solve for the potential energy as a 3D array of data for each electrode individually
    solve_trap_electrodynamics(trap, RELAXATION);

    // perform ion transport experiments now! we don't know how we will actually structure this part, just an example
    for (int e = 0; e < 5; e++)
    {
        // cost_experiment_e = perform_experiment(trap, e);
        // cost += cost_experiment_e;
    }

    cost += 0.5;

    return cost;
}

// main optimization loop
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


    double min_cost = 99999;
    const int dim = 12;

    // nlopt setup
    printf("- Constructing optimization problem...\n");
    nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA, dim);

    printf("Setting constraints...\n");
    // @TODO - constraints

    printf("Setting bounds...\n");
    // @TODO - bounds

    printf("Setting tolerances...\n");
    // @TODO - tolerances

    // generate initial guess trap from file
    printf("Constructing initial guess...\n");
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

    printf("Optimizing...\n");
    if (nlopt_optimize(opt, x, &min_cost) < 0)
    {
        printf("nlopt failed!\n");
    }
    else
    {
        printf("found minimum at f(");
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

