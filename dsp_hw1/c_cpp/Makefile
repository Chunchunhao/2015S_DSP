.PHONY: all clean

#CFLAGS=-std=c++11\
CXX=g++
CXXFLAGS=-std=c++0x
#LDFLAGS=-lm     # link to math library

TARGET1=train.cpp hmm.cpp
TARGET2=test.cpp hmm.cpp

all:
	$(CXX) $(CXXFLAGS) $(TARGET1) -o train
	$(CXX) $(CXXFLAGS) $(TARGET2) -o test

clean:
	$(RM) $(TARGET1) $(TARGET2)   # type make clean to remove the compiled file
