.PHONY: all train clean

CFLAGS+=
LDFLAGS+=-lm     # link to math library

CXX=clang++
FLAGS=-std=c++11 -stdlib=libc++
TRAIN=train
TRAINITR=5
INCLU=hmm
TEST=test

all: 
	$(CXX) $(FLAGS) $(TRAIN).cpp $(INCLU).cpp -o $(TRAIN)
	$(CXX) $(FLAGS) $(TEST).cpp $(INCLU).cpp -o $(TEST)
# type make/make all to compile test_hmm

run: 
	for nu in 1 2 3 4 5 ; do \
		./$(TRAIN) $(TRAINITR) ../model_init.txt ../seq_model_0$$nu.txt model_0$$nu.txt ; \
	done
	./$(TEST) ../modellist.txt ../testing_data1.txt result.txt; \

clean:
	$(RM) $(TRAIN)   # type make clean to remove the compiled file
