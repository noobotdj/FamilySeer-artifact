# Get started 
This tutorial contains examples of the following auto-tuning framework.

 - XLA
 - AutoTVM
 - Ansor
 - FamilySeer

# Requirements
You may need to install the following package to run this example.
```
pip3 install mxnet==1.8.0 torch==1.8.0 torchvision==0.9.0 transformers==4.11.2 vit-pytorch==0.24.3 huggingface-hub==0.0.18 onnx==1.10.1 gluonnlp==0.10.0
```
If you want to tune GPT2, you need to download the onnx model of GPT2 from [here](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.onnx). 
Then run the following command line:
```
mkdir onnx_models
mv gpt2-10.onnx onnx_models/
```
# Example
Since auto-tuning needs hours or days to find a good result, we prepare pre-tuned data so you can run directly without tuning. But you can still try tuning your model in your machine using the following command line.

__Run with pre-tuned data__
```
./run_with_pre-tuned.sh
```
The script will create the `evaluation_result` folder, and the result will be stored in this folder. By default, the script will run all eight models mentioned in the paper. You can change `eval_list` to run different models.

If you want to run XLA, please go to `./XLA/`.

__Run auto-tuning in your machine__

Run an example of FamilySeer:
```
//For GPU
python3 tune_network_gpu.py --model mobilenetv2_0.5 --gpu_num 2 --tune

//For CPU
python3 tune_network_x86.py --model mobilenetv2_0.5 --gpu_num 2 --tune
```
You can change `--gpu_num` number according to the GPUs you have.

Run an example of Ansor:
```
//For GPU
python3 tune_network_gpu.py --model mobilenetv2_0.5  --tune --mode ansor

//For CPU
python3 tune_network_x86.py --model mobilenetv2_0.5  --tune --mode ansor
```

Run an example of AutoTVM:
```
//For GPU
python3 tune_relay_gpu.py --model mobilenetv2_0.5  --tune

//For CPU
python3 tune_relay_x86.py --model mobilenetv2_0.5  --tune
```

If you want to try other models, change the `--model`. The supported model can be found in [TVM tutorials](https://tvm.apache.org/docs/how_to/compile_models/index.html).
