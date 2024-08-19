#!/bin/bash

out_folder="problems/seq-150-250"
base_problem="problems/ESC63.sop"
step=10
start=150
end=250

base_name=$(basename $base_problem .sop)
for n in $(seq $start $step $end)
do
	name="$base_name-scale-$n.sop"
	python evaluation/upscale_problem.py -s $n -o "$out_folder/$name" "$base_problem"
	echo "Saved scaled version $name"
done
