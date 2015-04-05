#!/bin/sh

a=1

while [ $a -lt 150 ]
do
	echo "Iteration : " [ $a ]
	b=1
	while [ $b -lt 6 ]
	do
		./train $a ../model_init.txt ../seq_model_0$b.txt model_0$b.txt
		b=`expr $b + 1`
	done
	./test ../modellist.txt ../testing_data1.txt result.txt ../testing_answer.txt
	a=`expr $a + 1`
done
