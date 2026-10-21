#include <stdlib.h>
#include "defs.h"
#include "data_structures.h"

void graph2layer(struct CircuitGraph instrgraph, int numions) //given the instructions find first set of instructions that can be executed in a single layer
{
    int *ionhashtable = (int*) malloc(sizeof(int)*numions); //count ion appearances, no entry should be greater than one per layer
    for (int i = 0; i < numions; i++) {
        ionhashtable[i] = 0;
    }

    struct InstructionNode* headnode = instrgraph.instruction_nodes;
    struct InstructionNode* currnode; 
    int currtargetion;
    int instrioncount = 1; //(temporary) might have to change InstructionNode struct to include number of ions being operated on
    int numnextnodes = 1; 
    int validinstrcount = 1; //for counting array size to return

    for (int i = 0; i < instrioncount; i++) {
        ionhashtable[headnode->target_ions[i]]++;
    }
    for (int i = 0; i < numnextnodes; i++) {
        currnode = headnode->prev_nodes + i;
        for (int i = 0; i < instrioncount; i++) {
	    currtargetion = currnode->target_ions[i];
	    if (ionhashtable[currnode->target_ions[i]] == 0) {
                ionhashtable[currnode->target_ions[i]]++;
		validinstrcount++;
                //add some way to save this node, maybe add to a linked list and traverse it to get array elements
	    } 
	    
        }
    }
}
