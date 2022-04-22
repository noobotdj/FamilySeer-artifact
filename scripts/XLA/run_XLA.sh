#!/bin/bash
mkdir evaluate_result
echo '============================='
echo 'XLA Performance Test on GPU'
for i in $(cat ./eval_list)
do
    echo $i
    ~/python3.8/bin/python3.8 -u tf2xla.py -d gpu \
        $i \
        1>./evaluate_result/[$i]\_[NVIDIA_V100]_XLA_B1.output
done

echo '============================='
echo 'XLA Performance Test on Silver_4210'
for i in $(cat ./eval_list)
do
    echo $i
    ~/python3.8/bin/python3.8 -u tf2xla.py \
        $i \
        1>./evaluate_result/[$i]\_[Silver_4210]_XLA_B1.output
done


