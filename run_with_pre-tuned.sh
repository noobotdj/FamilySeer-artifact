#!/bin/bash
mkdir evaluate_result
echo '============================='
echo 'Ansor Performance'
for i in $(cat ./modellist)
do
    echo $i
    ~/python3.8/bin/python3.8 -u scripts/tune_network_gpu.py --mode ansor --pre_tuned ./pre-tuned_data/Ansor/NVIDIA_V100/ --model $i 1>./evaluate_result/[$i]\_[NVIDIA_V100]_Ansor_B1.output
done


echo '============================='
echo 'FamilySeer Performance'
for i in $(cat ./modellist)
do
    echo $i
    ~/python3.8/bin/python3.8 -u scripts/tune_network_gpu.py --pre_tuned ./pre-tuned_data/FamilySeer/NVIDIA_V100/ --model $i 1>./evaluate_result/[$i]\_[NVIDIA_V100]_FamilySeer_B1.output
done

