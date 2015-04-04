#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "hmm.h"
#include <math.h>

using namespace std;

int main(int argc, char* argv[])
{
	if( argc != 4 ) {
		cerr << "./test modellist.txt testing_data1.txt result.txt" << endl;
		exit(1);
	}
	string fn_modelList(argv[1]), fn_testData(argv[2]), fn_result(argv[3]);

	vector<HMM> modelLists;
	int modelNum = 0;
	modelNum = load_models(modelLists, fn_modelList);

/*
	cout << "Total modes = " << modelNum << endl;

	for( int i=0; i<modelNum; i++ )
		modelLists[i].verifyHMM();
*/
	testing(modelLists, fn_testData, fn_result);
	cout << "Finish Testing" << endl;
	
	
	string test_answer("../testing_answer.txt");
	double acc = accuracy(fn_result, test_answer);
	cout << "accuracy = " << acc << endl;
	return 0;
}



/**   test.cpp will 
Open and Read File 
- modellist.txt
--- model_01{~05}.txt
- testing_data.txt

Generate file 
 - result.txt
 - acc.txt

**/ 