#!/bin/bash

mapfile -t variants < <(./build/main -l)
#variants=("gpumax" "neighbor" "localant")
problems=("./problems/ESC78.sop")
runs=1

for n in $(seq $runs)
do
	echo "Starting run #$n"
	for problem in $problems
	do
		name=$(basename $problem .sop)
		for variant in "${variants[@]}"
		do
			if ./build/main $problem -c $variant -r 500 -a -o "./evaluation/$name.profile.csv";
			then
				echo "Finished $variant on $name"
			else
				echo "Variant $variant returned error code on problem $name ($problem)"
				echo "Ignoring error"
			fi
		done
	done
done
