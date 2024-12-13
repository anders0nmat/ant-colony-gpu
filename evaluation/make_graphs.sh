#!/bin/bash

python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'sequential' on selected problems" \
	-o evaluation/figures/sequential.png \
	evaluation/sequential.profile.csv
python evaluation/plot.py \
	-pl \
	-w 0.25 \
	-t "Logarithmic execution time of variant 'sequential' on selected problems" \
	-o evaluation/figures/sequential-log.png \
	evaluation/sequential.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'manyant' on selected problems" \
	-o evaluation/figures/manyant.png \
	evaluation/manyant.profile.csv
python evaluation/plot.py \
	-pl \
	-w 0.25 \
	-t "Logarithmic execution time of variant 'manyant' on selected problems" \
	-o evaluation/figures/manyant-log.png \
	evaluation/manyant.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'sequential' and 'manyant' on selected problems" \
	-o evaluation/figures/sequential-manyant.png \
	evaluation/sequential.profile.csv \
	evaluation/manyant.profile.csv
python evaluation/plot.py \
	-pl \
	-w 0.25 \
	-t "Logarithmic execution time of variant 'sequential' and 'manyant' on selected problems" \
	-o evaluation/figures/sequential-manyant-log.png \
	evaluation/sequential.profile.csv \
	evaluation/manyant.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Logarithmic execution time of variant 'gpupher' and 'phercomp' on selected problems" \
	-o evaluation/figures/gpupher-phercomp.png \
	evaluation/gpupher.profile.csv \
	evaluation/phercomp.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of program part 'update' for variants 'manyant' and 'gpupher'" \
	-o evaluation/figures/manyant-gpupher-upda.png \
	evaluation/manyant.profile.csv \
	evaluation/gpupher.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'gpupher' and 'phercomp' on selected problems" \
	-o evaluation/figures/gpupher-phercomp.png \
	evaluation/gpupher.profile.csv \
	evaluation/phercomp.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'phercomp' and 'binsearch' on selected problems" \
	-o evaluation/figures/phercomp-binsearch.png \
	evaluation/phercomp.profile.csv \
	evaluation/binsearch.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'binsearch' and 'depmask' on selected problems" \
	-o evaluation/figures/binsearch-depmask.png \
	evaluation/binsearch-depmask.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'depmask' and 'samplemask' on selected problems" \
	-o evaluation/figures/depmask-samplemask.png \
	evaluation/depmask-samplemask.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'depmask' on increasing problem sizes" \
	-o evaluation/figures/depmask-step.png \
	evaluation/depmask-step-test.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variants 'depmask', 'samplemask' and 'parant' on selected problems" \
	-o evaluation/figures/depmask-samplemask-parant.png \
	evaluation/depmask-samplemask-parant.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'parant' on increasing problem sizes" \
	-o evaluation/figures/parant-seq.png \
	evaluation/parant-seq.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.25 \
	-t "Execution time of variant 'depmask' and 'samplemask on increasing problem sizes" \
	-o evaluation/figures/depmask-samplemask-seq.png \
	evaluation/depmask-samplemask-step.profile.csv
python evaluation/plot.py \
	-p \
	-w 0.2 \
	evaluation/parantN-big-step.profile.csv \
	--order ESC63s400.sop ESC63s500.sop ESC63s600.sop ESC63s700.sop ESC63s800.sop ESC63s900.sop ESC63s1000.sop
	-o evaluation/figures/parantN-big-step-nice.png