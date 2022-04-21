from sample import SequenceGenerator
import click
import time
import os, argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA
@click.command()
@click.option('--model-path', type=str, default="./model", show_default=True, help="Model Path")
@click.option('--model-param', type=str, default="./model/model_par.json", show_default=True, help="Model Parm")
@click.option('--vocab', type=str, default="./data/bpe_model.model", show_default=True, help="Vocab")
@click.option('--seq-len', type=int, default=512, show_default=True, help="seq_len")
@click.option('--temperature', type=float, default=1.0, show_default=True, help="seq_len")
@click.option('--top-k', type=int, default=8, show_default=True, help="seq_len")
@click.option('--top-p', type=float, default=0.9, show_default=True, help="seq_len")
@click.option('--nucleus_sampling', type=bool, default=False, show_default=True, help="seq_len")
@click.option('--context', type=str, default="sample context", show_default=True, help="Context given to model")
def seq_gen(model_path, model_param, vocab, seq_len, temperature, top_k, top_p, nucleus_sampling, context):
	sg = SequenceGenerator(model_path, model_param, vocab)
	sg.load_weights()
	nSteps=25
	avg_time=0
	for i in range(0, nSteps):
		time1 = time.time()
		generated_seq = sg.sample_sequence(context,
									   seq_len=seq_len,
									   temperature=temperature,
									   top_k=top_k,
									   top_p=top_p,
									   nucleus_sampling=nucleus_sampling)
		time2 = time.time()
		if i < 5:
			continue
		avg_time += float(time2-time1)
		info = '-- %d, iteration time(ms) is %.4f' %(i, (float(time2-time1)*1000))
		print(info)
	
	avg_time = avg_time / (nSteps-5)
	name = "gpt2"
	print("@@ %s, average time(ms) is %.4f" % (name, avg_time*1000))
	print('FINISH')
	
	#print("Generated seq by model:- " + generated_seq)


if __name__ == "__main__":
	seq_gen()
