# Get started 
This tutorial contains an example of using FamilySeer and Ansor to train the model.

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
[guide-level-explanation]: #guide-level-explanation
Run an example of FamilySeer:
```
python3 tune_network_gpu.py --model mobilenetv2_0.5 --gpu_num 2 --tune
```
You can change `--gpu_num` number according to the GPUs you have.

Run an example of Ansor:
```
python3 tune_network_gpu.py --model mobilenetv2_0.5  --tune --mode ansor
```
If you want to try other models, change the `--model`. Supported model can be found in `./modellist`. You can also try you own model following the [TVM tutorials](https://tvm.apache.org/docs/how_to/compile_models/index.html).
