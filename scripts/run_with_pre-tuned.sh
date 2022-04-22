#!/bin/bash
mkdir evaluate_result

# evaluate GPU
echo '============================='
echo 'Ansor Performance Test on GPU'
for i in $(cat ./eval_list)
do
    echo $i
    #~/python3.8/bin/python3.8 -u tune_network_gpu.py --mode ansor --pre_tuned ../pre-tuned_data/Ansor/NVIDIA_V100/ --model $i 1>./evaluate_result/[$i]\_[NVIDIA_V100]_Ansor_B1.output
done


echo '============================='
echo 'FamilySeer Performance Test on GPU'
for i in $(cat ./eval_list)
do
    echo $i
    #~/python3.8/bin/python3.8 -u tune_network_gpu.py --pre_tuned ../pre-tuned_data/FamilySeer/NVIDIA_V100/ --model $i 1>./evaluate_result/[$i]\_[NVIDIA_V100]_FamilySeer_B1.output
done

echo '============================='
echo 'AutoTVM Performance Test on GPU'
for i in $(cat ./eval_list)
do
    echo $i
    #~/python3.8/bin/python3.8 -u tune_relay_gpu.py --pre_tuned ../pre-tuned_data/AutoTVM/NVIDIA_V100/ --model $i 1>./evaluate_result/[$i]\_[NVIDIA_V100]_AutoTVM_B1.output
done


# evaluate CPU
echo '============================='
echo 'Ansor Performance Test on CPU'
for i in $(cat ./eval_list)
do
    echo $i'''
    ~/python3.8/bin/python3.8 -u tune_network_x86.py --mode ansor \
    --pre_tuned ../pre-tuned_data/Ansor/Silver_4210/ \
    --model $i \
    1>./evaluate_result/[$i]\_[Silver_4210]_Ansor_B1.output'''
done


echo '============================='
echo 'FamilySeer Performance Test on CPU'
for i in $(cat ./eval_list)
do
    echo $i'''
    ~/python3.8/bin/python3.8 -u tune_network_x86.py \
    --pre_tuned ../pre-tuned_data/FamilySeer/Silver_4210/ \
    --model $i \
    1>./evaluate_result/[$i]\_[Silver_4210]_FamilySeer_B1.output'''
done

echo '============================='
echo 'AutoTVM Performance Test on CPU'
for i in $(cat ./eval_list)
do
    echo $i
    ~/python3.8/bin/python3.8 -u tune_relay_x86.py \
    --pre_tuned ../pre-tuned_data/AutoTVM/Silver_4210/ \
    --model $i \
    1>./evaluate_result/[$i]\_[Silver_4210]_AutoTVM_B1.output
done
