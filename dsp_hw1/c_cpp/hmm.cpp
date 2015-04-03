
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

#include "hmm.h"

using namespace std;


HMM::HMM(string& init_model) {
	loadHMM(init_model);
}

void HMM::loadHMM(string& fn_init_model) 
{
	ifstream init_model;
	open_file(init_model, fn_init_model);
	string line; // each line ;
	while( getline(init_model, line) ) {
		istringstream iss(line);
		string model_info;
		int model_parameter;
		iss >> model_info >> model_parameter;

		if( model_info == "initial:" ){
			// cout << "initial: " << model_info << model_parameter << endl;
			state_num = model_parameter;
			initial.resize(state_num);
			getline(init_model, line );
			istringstream iss_sn(line);
			for( int i=0; i<state_num; i++) {
				iss_sn >> initial[i];
			}

		}
		else if( model_info == "transition:" ) {
			// cout << "transition: " << model_info << model_parameter << endl;
			state_num = model_parameter;
			transition.resize(state_num);
			for( int i=0; i<state_num; i++) {
				transition[i].resize(state_num);
			}

			for( int i=0; i<state_num; i++) {
				getline(init_model, line);
				istringstream iss_st(line);
				for( int j=0; j<state_num; j++ ) {
					iss_st >> transition[i][j];
				}
			}
		}
		else if( model_info == "observation:" ) {
			// cout << "observation: " << model_info << model_parameter << endl;
			observe_num = model_parameter;
			observation.resize(observe_num);
			for( int i=0; i<observe_num; i++) {
				observation[i].resize(state_num);
			}

			for( int i=0; i<observe_num; i++) {
				getline(init_model, line);
				istringstream iss_ob(line);
				for( int j=0; j<state_num; j++ ) {
					iss_ob >> observation[i][j];
				}
			}
		}
		else {
			continue;
		}
	}

	init_model.close();
	init_model.clear();
}

void HMM::dumpHMM(string& fn_hmm_dump) 
{
	ofstream dump_hmm;
	write_file(dump_hmm, fn_hmm_dump);
	dump_hmm << "initial: " << state_num << endl;
	for( int i=0; i< state_num - 1; i++)
		dump_hmm << setprecision(5) <<  initial[i] << " ";
	dump_hmm << setprecision(5) <<  initial[state_num-1] << endl;

	dump_hmm << endl << "transition: " << state_num << endl;
	for( int i=0; i< state_num; i++) {
		for( int j=0; j< state_num-1; j++) {
			dump_hmm << setprecision(5) <<  transition[i][j] << " ";
		}
		dump_hmm << setprecision(5) <<  transition[i][state_num-1] << endl;
	}

	dump_hmm << endl << "observation: " << observe_num << endl;
	for( int i=0; i< observe_num; i++) {
		for( int j=0; j< state_num-1; j++) {
			dump_hmm << setprecision(5) <<  observation[i][j] << " ";
		}
		dump_hmm << setprecision(5) <<  observation[i][state_num-1] << endl;
	}
}

void HMM::BWA(int iteration, string& fn_seq_model) {

  /// -- Traning iteration 
  while( --iteration ) {
  cout << iteration << " training left " << endl;
	
	ifstream samples;
	open_file( samples, fn_seq_model);
	string sampleLine;

	///--- Store accumulate variable
	int numberOfSample = 0;
	
	// accumulate samples' initial 
	vector<double> accm_initial;
	accm_initial.resize(state_num);

	// accumulate sample's transition
	vector< vector<double> > accm_transition_nu; // numerator ( up)
	accm_transition_nu.resize(state_num);
	for( int i=0; i<state_num; i++ ) 
		accm_transition_nu[i].resize(state_num);	
	vector< vector<double> > accm_transition_de; // denominator ( down)
	accm_transition_de.resize(state_num);
	for( int i=0; i<state_num; i++ ) 
		accm_transition_de[i].resize(state_num);

	// accmulate samples' observation
	vector< vector<double> > accm_observation_nu; // numerator ( up)
	accm_observation_nu.resize(observe_num);
	for( int i=0; i<observe_num; i++)
		accm_observation_nu[i].resize(state_num);
	vector< vector<double> > accm_observation_de; // denominator ( down)
	accm_observation_de.resize(observe_num);
	for( int i=0; i<observe_num; i++)
		accm_observation_de[i].resize(state_num);


	while( getline(samples, sampleLine) ) {
		vector< int > sample;
		sample = samplize( sampleLine );
		int sampleLen = sample.size();

		/// O --- Alpha : Forward Aldorithm 
		vector< vector<double> > alpha;
		alpha.resize(sampleLen);
		for( int t=0; t<sampleLen; t++ )
			alpha[t].resize(state_num);

		// Alpha : Initialization
		for( int i=0; i<state_num; i++ )
			alpha[0][i] = initial[i] * observation[sample[0]][i];		

		// Alpha : Induction 
		for( int t=1; t<sampleLen; t++ ) {			
			for( int i=0; i<state_num; i++ ){
				double prevSumAlpha = 0.0;
				for( int k=0; k<state_num; k++) {
					prevSumAlpha += alpha[t-1][k] * transition[k][i];
				}
				alpha[t][i] = prevSumAlpha * observation[sample[t]][i];
			}
		}

		/// O--- Beta
		vector< vector<double> > beta;
		beta.resize(sampleLen);
		for( int t=0; t<sampleLen; t++ )
			beta[t].resize(state_num);
		
		// Beta : initialization
		for( int i=0; i<state_num; i++ )
			beta[sampleLen-1][i] = 1.0;

		// Beta : Induction
		for( int t=sampleLen-2; t>=0; t-- ) {
			for( int i=0; i<state_num; i++ ) {
				double postSumBeta = 0.0;
				for( int j=0; j<state_num; j++) {
					postSumBeta += transition[i][j] * observation[sample[t+1]][j] * beta[t+1][j];
				}
				beta[t][i] = postSumBeta;
			}
		}

		/// O--- Gemma
		vector< vector<double> > gemma;
		gemma.resize(sampleLen);
		for( int t=0; t<sampleLen; t++ )
			gemma[t].resize(state_num);

		for( int t=0; t<sampleLen; t++ ) {
			double normalize_gemma = 0.0;
			for( int i=0; i<state_num; i++ ) {
				normalize_gemma += alpha[t][i] * beta[t][i];
			}
			for( int i=0; i<state_num; i++ ) {
				gemma[t][i] = alpha[t][i] * beta[t][i] / normalize_gemma;
			}
		}
		
		/// O--- epsilon
		vector< vector< vector<double> > > epsilon;
		epsilon.resize(sampleLen); // time * state * state 
		for( int i=0; i<sampleLen; i++ ) {
			epsilon[i].resize(state_num);
			for( int j=0; j<state_num; j++ ) {
				epsilon[i][j].resize(state_num);
			}
		}

		for( int t=0; t<sampleLen-1; t++ ) {
			double normalize_epsilon = 0.0;
			for( int i=0; i<state_num; i++ ) {
				for( int j=0; j<state_num; j++) {
					normalize_epsilon += alpha[t][i] * transition[i][j] * observation[sample[t+1]][j] * beta[t+1][j];
				}
			}
			for( int i=0; i<state_num; i++ ) {
				for( int j=0; j<state_num; j++) {
					epsilon[t][i][j] = alpha[t][i] * transition[i][j] * observation[sample[t+1]][j] * beta[t+1][j] / normalize_epsilon;
				}
			}
		}


		/// O--- Update Accumlate
		++ numberOfSample;

		// Accumlate initial 
		for( int i=0; i<state_num; i++ ) {
			accm_initial[i] += gemma[0][i];
		}


		// Accumlate Transition 
		for( int i=0; i<state_num; i++ ) {
			for( int j=0; j<state_num; j++) {
				double epsilon_sum = 0.0;
				double gemma_sum = 0.0;
				for( int t=0; t<sampleLen-1; t++ ) {
					epsilon_sum += epsilon[t][i][j];
					gemma_sum += gemma[t][i];
				}
				
				accm_transition_nu[i][j] += epsilon_sum;
				accm_transition_de[i][j] += gemma_sum;
			}
		}

		// Accumulate Observation 
		for( int j=0; j<observe_num; j++ ) {
			for( int k=0; k<state_num; k++) {
				double 	gemma_sum_with_observ = 0.0;
				double  gemma_sum = 0.0;
				for( int t=0; t<sampleLen; t++) {
					if( sample[t] == k ) {
						gemma_sum_with_observ += gemma[t][j];
					}
					gemma_sum += gemma[t][j];
				}
				
				accm_observation_nu[j][k] = gemma_sum_with_observ;
				accm_observation_de[j][k] = gemma_sum;
			}
		}

	} // end of reading sample 


	/// --- Update HMM
	cout << "Total input is " << numberOfSample << endl;
	// update initial 
	for( int i=0; i<state_num; i++)
		initial[i] = preciFive(accm_initial[i] / numberOfSample );
	// update transition
	for( int i=0; i<state_num; i++)
		for( int j=0; j<state_num; j++)
			transition[i][j] = preciFive(accm_transition_nu[i][j] / accm_observation_de[i][j] / numberOfSample );
	// update observation
	for( int j=0; j<observe_num; j++)
		for( int k=0; k<state_num; k++)
			observation[j][k] = preciFive(accm_transition_nu[j][k] / accm_observation_de[j][k] / numberOfSample );

	cout << "Finish Training" << endl;
	verifyHMM();
	justifyHMM();
	verifyHMM();
	samples.close();
  }
}

void HMM::verifyHMM() {
	double accum = 0.0;
	// check initial
	for( int i=0; i<state_num; i++) {
		accum += initial[i];
	}
	if( accum != 1.0 ){
		cerr << "Iniitial (pi) is not equal to 1 , is { " << accum << " }" << endl;
	}

	// check transition
	for( int i=0; i<state_num; i++ ) {
		accum = 0.0;
		for( int j=0; j<state_num; j++) {
			accum += transition[i][j];
		}
		if( accum != 1.0 ) {
			cerr << "Transition row [ " << i << " ] is not equal to 1 , is { " << accum << " }"  << endl;			
		}
	}

	// check obervation
	for( int k=0; k<state_num; k++ ) {
		accum = 0.0;
		for( int j=0; j<observe_num; j++ ) {
			accum += observation[j][k];
		}
		if( accum != 1.0 ) {
			cerr << "Observation column [ " << k << " ] is not equal to 1 , is { " << accum << " }"  << endl;			
		}
	}

}

void HMM::justifyHMM() {

	double delta = 0.0, shared=0.0;
	// check initial
	for( int i=0; i<state_num; i++) {
		delta += initial[i];
	}
	if( delta != 1.0 ){
		cout << "Initial shared delta : " << delta << endl;
		shared = (delta-1) / state_num;
		for( int i=0; i<state_num; i++) {
			initial[i] += shared;
		}
	}

	// check transition
	for( int i=0; i<state_num; i++ ) {
		delta = 0.0;
		for( int j=0; j<state_num; j++) {
			delta += transition[i][j];
		}
		if( delta != 1.0 ) {
			cout << "Transition shared delta : " << delta << endl;
			shared = (delta-1) / state_num;
			for( int j=0; j<state_num; j++) {
				transition[i][j] += shared;
			}
		}
	}

	// check obervation
	for( int k=0; k<state_num; k++ ) {
		delta = 0.0;
		for( int j=0; j<observe_num; j++ ) {
			delta += observation[j][k];
		}
		if( delta != 1.0 ) {
			cout << "Observation shared delta : " << delta << endl;
			shared = (delta-1) / state_num;
			for( int j=0; j<observe_num; j++) {
				observation[j][k] += shared;
			}
		}
	}
}

bool cmpDouble (double A, double B) {
	double diff = A - B ;
	double lamda = 0.00001;
	return (diff - lamda) && (-diff - lamda );
}


// There are something happend if we estimate under three precision they have validation problem
double preciFive ( double in) {
	double result;
	int trans = in * 100000;
	if( (trans % 10) > 4 ) {
		trans = trans - ( trans % 10 ) + 10;
	}
	else {
		trans = trans - ( trans % 10 );
	}
	result = (double)trans / 100000;
	return result;
}

vector<int> samplize( string& samples ) {
	vector<int> parseSample2int;
	for( char c: samples )
		if( c >= 'A' && c <= 'Z') {
			parseSample2int.push_back(c - 'A');
		}
		else {
			cerr << "Something wrong in seq_model!" << endl;
			exit(1);
		} 
	return parseSample2int;
}

ifstream& open_file (ifstream& in, const string& filename)
{
  in.close();
  in.clear();
  in.open(filename.c_str());
  if ( !in ){
  	cerr << "Open File { " << filename << " } FAILED ! " << endl;
  	exit(1);
  }
  return in;
}


ofstream& write_file (ofstream& out, const string& filename)
{
  out.close();
  out.clear();
  out.open(filename.c_str(), ios::trunc);
  if ( !out ){
  	cerr << "Open File { " << filename << " } FAILED ! " << endl;
  	exit(1);
  }
  return out;
}



