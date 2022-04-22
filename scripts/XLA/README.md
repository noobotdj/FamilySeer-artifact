# Get started 
This tutorial contains an example of using Tensorflow XLA to evaluate the model.

# Requirements
You may need to install the following package to run this example.
```
pip3 install tensorflow==2.6.0 tensorflow-text==2.6.0 tf-sentencepiece==0.1.90 tensorflow-addons==0.16.1 tensorflow-hub==0.12.0
```

# Example
Run XLA with script:
```
./run_XLA.sh
```
The script will create a `evaluation_result` folder and the result will be stored in this folder. 

XLA requires models from Tensorflow Hub and only six of them can be found on Tensorflow Hub, we find GPT2 and ViT-Huge from other repos, please follow the command line to test GPT2 and ViT-Huge.
```
//GPT2
cp sequence_generator.py gpt-2-tensorflow2.0/
cd gpt-2-tensorflow2.0/
python3 train_gpt2.py
python3 sequence_generator.py

//ViT-Huge (Before doing this, return to the ./XLA folder)
cp train.py vit/
cd vit/
python3 train.py
```