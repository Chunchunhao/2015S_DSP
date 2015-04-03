#ifndef MODELIO_H
#define MODELIO_H

#include <string>
#include <fstream>

class HMM 
{

public:
	HMM() = default;
	HMM(std::string&  );	// load HMM;

	void loadHMM(std::string& );
	void dumpHMM(std::string& );
	// int  loadHMM
	// void dumpHMM

	void BWA(int, std::string& );
	void verifyHMM();
	void justifyHMM();

private:
	std::string model_name;
	int state_num;									// number of state
	int observe_num; 								// number of observation 
	std::vector<double> initial;					// initial prob.
	std::vector< std::vector<double> > transition; 	// transition prob.
	std::vector< std::vector<double> > observation;	// observation prob.
};

bool cmpDouble (double , double );
double preciFive ( double );
std::vector<int> samplize( std::string& samples );
std::ifstream& open_file (std::ifstream& in, const std::string& filename); 
std::ofstream& write_file (std::ofstream& out, const std::string& filename);


// std::

#endif 
