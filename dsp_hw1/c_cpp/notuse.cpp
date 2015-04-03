/* 
// not found 
// #include <boost/format.hpp>

void HMM::dumpHMM(string& fn_hmm_dump) {
	ofstream dump_hmm;
	write_file(dump_hmm, fn_model_dump);
	dump_hmm << "initial: " << state_num << endl;
	for( int i=0; i< state_num - 1; i++)
		dump_hmm << boost::format("%.5lf") % initial[i] << " ";
	dump_hmm << boost::format("%.5lf") % initial[state_num-1] << endl;

	dump_hmm << endl << "transition: " << state_num << endl;
	for( int i=0; i< state_num; i++) {
		for( int j=0; j< state_num-1; j++) {
			dump_hmm << boost::format("%.5lf") % transition[i][j] << " ";
		}
		dump_hmm << boost::format("%.5lf") % transition[i][state_num-1] << endl;
	}

	dump_hmm << endl << "observation: " << observe_num << endl;
	for( int i=0; i< observe_num; i++) {
		for( int j=0; j< observe_num-1; j++) {
			dump_hmm << boost::format("%.5lf") % observation[i][j] << " ";
		}
		dump_hmm << boost::format("%.5lf") % observation[i][observe_num-1] << endl;
	}
	
}
*/ 



