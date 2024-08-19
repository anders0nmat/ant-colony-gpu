#!/bin/bash

#mapfile -t variants < <(./build/main -l)
#variants=("depmask" "samplemask")
variants=("parant")
#problems=("./problems/ESC11.sop" "./problems/ESC25.sop" "./problems/ESC47.sop" "./problems/prob.100.sop")
#problems=("problems/rbg109a.sop" "problems/rbg174a.sop")
#problems=("problems/rbg253a.sop" "problems/rbg323a.sop")
#problems=("./problems/ESC47.sop" "problems/rbg109a.sop" "problems/rbg174a.sop" "problems/rbg253a.sop" "problems/rbg323a.sop")
problems=(./problems/seq-250-350/*.sop)
runs=10

for n in $(seq $runs)
do
	echo "Starting run #$n"
	for problem in "${problems[@]}"
	do
		#name=$(basename $problem .sop)
		for variant in "${variants[@]}"
		do
			#name="$variant"
			name="parant-250"
			if ./build/main $problem -c $variant -r 500 -a -o "./evaluation/$name.profile.csv";
			then
				echo "Finished $variant on $problem"
			else
				echo "Variant $variant returned error code on problem $name ($problem)"
				echo "Ignoring error"
			fi
		done
	done
done
