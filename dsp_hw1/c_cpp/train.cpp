#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "hmm.h"

using namespace std;

/**
Read:
- model_init.txt
- seq_model_01.txt

Genearte 
- model_01.txt


**/

int main (int argc, char* argv[])
{
	if( argc != 5 ) {
		cerr << "./train iteration model_init.txt seq_model_0X.txt model_0X.txt" << endl;
		exit(1);
	}
	// cout << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << endl;
	
	int epoch = atoi(argv[1]); // number of iteration
	cout << "Training iteration: " << epoch << endl;
	ifstream model_init, seq_model, dump_model;
	string fn_hmm_init(argv[2]), fn_seq_model(argv[3]), fn_dump_hmm(argv[4]);

	auto model = HMM( fn_hmm_init, fn_dump_hmm);
	model.BWA( epoch, fn_seq_model);
	model.dumpHMM( fn_dump_hmm);
	model.verifyHMM();

	return 0;

}



