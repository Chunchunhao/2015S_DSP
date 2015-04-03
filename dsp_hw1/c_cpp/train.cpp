#include <iostream>
#include <fstream>

#include <string>
#include <vector>

#include <cstdlib>

#include "hmm.h"

using namespace std;


// argv[0]

int main (int argc, char* argv[])
{
	if( argc != 5 ) {
		cerr << "./train iteration model_init.txt seq_model_0X.txt model_0X.txt" << endl;
		exit(1);
	}
	else {
		cout << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << endl;
	}


	int epoch = atoi(argv[1]); // number of iteration
	cout << "Training iteration: " << epoch << endl;

	ifstream model_init, seq_model, dump_model;
	string fn_hmm_init(argv[2]), fn_seq_model(argv[3]), fn_dump_hmm(argv[4]);
	fn_hmm_init = "../" + fn_hmm_init;
	fn_seq_model = "../" + fn_seq_model;


	auto model = HMM( fn_hmm_init );
	// model.verifyHMM();
	model.BWA( epoch, fn_seq_model);

	model.dumpHMM( fn_dump_hmm);
	// model.verifyHMM();


/*	
	HMM hmms[5];
	load_models( "modellist.txt", hmms, 5);
	dump_models( hmms, 5);
	HMM hmm_initial;
	loadHMM( &hmm_initial, "../model_init.txt" );
	dumpHMM( stderr, &hmm_initial );
*/

	return 0;

}



/**
Read:
- model_init.txt
- seq_model_01.txt

Genearte 
- model_01.txt


**/

