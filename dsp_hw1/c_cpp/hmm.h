#ifndef MODELIO_H
#define MODELIO_H

#include <string>
#include <fstream>

class HMM 
{

public:
	HMM() = default;
	HMM(std::string&, std::string&);	// load HMM;
	HMM(const HMM &);	// copy constructor
	~HMM();	// deconstructor

	void loadHMM(std::string&, std::string&);
	void dumpHMM(std::string& );

	double viterbi(const std::vector<int> &);

	void BWA(int, std::string& );
	void verifyHMM();
	void justifyHMM();

	std::string model_name;
	
private:
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

int load_models( std::vector< HMM > &, std::string );
void testing( std::vector<HMM> &, std::string, std::string );

double accuracy(std::string, std::string);
#endif 
