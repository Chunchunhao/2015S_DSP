.PHONY: all train clean

CFLAGS+=
LDFLAGS+=-lm     # link to math library

CXX=clang++
FLAGS=-std=c++11 -stdlib=libc++
TRAIN=train
TRAINITR=10
INCLU=hmm


all: 
	$(CXX) $(FLAGS) $(TRAIN).cpp $(INCLU).cpp -o $(TRAIN)
# type make/make all to compile test_hmm

train: 
	for nu in 1 2 3 4 5 ; do \
		./$(TRAIN) $(TRAINITR) model_init.txt seq_model_0$$nu.txt model_0$$nu.txt ; \
	done
clean:
	$(RM) $(TRAIN)   # type make clean to remove the compiled file
