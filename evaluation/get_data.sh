#!/bin/bash

#mapfile -t variants < <(./build/main -l)
#variants=("localant" "gpumax")
variants=("manyant")
#variants=("depmask" "samplemask")
#variants=("parant" "parant2" "parant3" "parant4")
#variants=("parant4" "localant" "gpumax")
#problems=("./problems/ESC11.sop" "./problems/ESC25.sop" "./problems/ESC47.sop" "./problems/prob.100.sop")
#problems=("problems/rbg109a.sop" "problems/rbg174a.sop")
#problems=("problems/rbg253a.sop" "problems/rbg323a.sop")
#problems=("./problems/ESC47.sop" "problems/rbg109a.sop" "problems/rbg174a.sop" "problems/rbg253a.sop" "problems/rbg323a.sop")
#problems=(./problems/rbg-seq-150-300/*.sop)
#problems=(./problems/seq-400-1000/*.sop)
#problems=(./problems/seq-150-250/*.sop ./problems/seq-250-350/*.sop ./problems/seq-400-1000/*.sop)
problems=(./problems/seq-250-350/*.sop)
#problems=(./problems/seq-150-250/*.sop)

runs=3

for n in $(seq $runs)
do
	echo "Starting run #$n"
	for problem in "${problems[@]}"
	do
		#name=$(basename $problem .sop)
		for variant in "${variants[@]}"
		do
			name="$variant.step.shower"
			#name="parantN-big-step"
			echo "Starting $variant on $problem"
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

