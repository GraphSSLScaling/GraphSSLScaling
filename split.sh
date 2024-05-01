#!/bin/bash

ratio=0.1

models=('GraphCL') 
datasets=('MUTAG')
template='singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task SGC --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --config_file random_config/graphcl'
commands=()

for i in $(seq 0 $ratio 1); do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            command="${template/MODEL_PLACEHOLDER/$model}"
            command="${command/DATASET_PLACEHOLDER/$dataset}"
            command="${command/RATIO_PLACEHOLDER/$i}"
            commands+=("$command")
        done
    done
done

for command in "${commands[@]}";do
    echo $command
done
parallel -j 1 eval ::: "${commands[@]}"