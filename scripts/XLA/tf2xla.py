import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import os, argparse
import tensorflow_text

def tf2xla(sm_path, nSteps=25):
    if sm_path == 'densenet121':
      model = tf.keras.applications.DenseNet121(weights='imagenet', classes=1000)
    elif sm_path == 'vgg16':
      model = tf.keras.applications.VGG16(weights='imagenet', classes=1000)
    elif sm_path == 'vgg19':
      model = tf.keras.applications.VGG19(weights='imagenet', classes=1000)

    elif sm_path == 'roberta_large':
      # define a text embedding model
      text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
      preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/roberta_en_cased_preprocess/1")
      encoder_inputs = preprocessor(text_input)

      encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/roberta_en_cased_L-24_H-1024_A-16/1", trainable=True)
      encoder_outputs = encoder(encoder_inputs)
      pooled_output = encoder_outputs["pooled_output"]      # [batch_size, 1024].
      sequence_output = encoder_outputs["sequence_output"]  # [batch_size, seq_length, 1024].

      model = tf.keras.Model(text_input, pooled_output)

      
      sentences = tf.constant(["hello"],dtype=tf.string)
      #sentences = sentences.gpu()
      avg_time=0
      with tf.device('CPU:0'):
        for i in range(0, nSteps):
          time1 = time.time()
          model(sentences)
          time2 = time.time()
          if i < 5:
            continue
          avg_time += float(time2-time1)
          info = '-- %d, iteration time(ms) is %.4f' %(i, (float(time2-time1)*1000))
          print(info)
        
        avg_time = avg_time / (nSteps-5)
        name = os.path.basename(sm_path)
        print("@@ %s, average time(ms) is %.4f" % (name, avg_time*1000))
        print('FINISH')
    elif sm_path == 'bert_large':
      # define a text embedding model
      text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
      preprocessor = hub.KerasLayer(
          "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
      encoder_inputs = preprocessor(text_input)
      encoder = hub.KerasLayer(
          "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/4",
          trainable=True)
      outputs = encoder(encoder_inputs)
      pooled_output = outputs["pooled_output"]      # [batch_size, 1024].
      sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 1024].

      embedding_model = tf.keras.Model(text_input, pooled_output)

      sentences = tf.constant(["hello"],dtype=tf.string)
      
      avg_time=0
      for i in range(0, nSteps):
        time1 = time.time()
        embedding_model(sentences)
        time2 = time.time()
        if i < 5:
          continue
        avg_time += float(time2-time1)
        info = '-- %d, iteration time(ms) is %.4f' %(i, (float(time2-time1)*1000))
        print(info)
      
      avg_time = avg_time / (nSteps-5)
      name = os.path.basename(sm_path)
      print("@@ %s, average time(ms) is %.4f" % (name, avg_time*1000))
      print('FINISH')
    elif sm_path=="vit":
      model = tf.keras.Sequential([
          hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b32_classification/1")
      ])
      shape=[1,224,224,3]
      images = np.ones(shape, dtype=np.float32)
      
      avg_time=0
      for i in range(0, nSteps):
        time1 = time.time()
        predictions = model.predict(images)
        time2 = time.time()
        if i < 5:
          continue
        avg_time += float(time2-time1)
        info = '-- %d, iteration time(ms) is %.4f' %(i, (float(time2-time1)*1000))
        print(info)
      
      avg_time = avg_time / (nSteps-5)
      name = os.path.basename(sm_path)
      print("@@ %s, average time(ms) is %.4f" % (name, avg_time*1000))
      print('FINISH')
    
    else:
      if sm_path == 'resnet50_v1':
        url_path = 'https://tfhub.dev/tensorflow/resnet_50/classification/1'
      elif sm_path == 'resnet152_v2':
        url_path = 'https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5'
      elif sm_path == 'mobilenet0.5':
        url_path = 'https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/5'
      elif sm_path == 'mobilenetv2_0.5':
        url_path = 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/5'
      else:
        url_path = 'https://tfhub.dev/tensorflow/resnet_50/classification/1'
      model = tf.keras.Sequential([
          hub.KerasLayer(url_path)
          ])
    
      shape=[1,224,224,3]
      picture = np.ones(shape, dtype=np.float32)
      
      avg_time=0
      for i in range(0, nSteps):
        time1 = time.time()
        ret = model.predict(picture, batch_size=1)
        time2 = time.time()
        if i < 5:
          continue
        avg_time += float(time2-time1)
        info = '-- %d, iteration time(ms) is %.4f' %(i, (float(time2-time1)*1000))
        print(info)
      
      avg_time = avg_time / (nSteps-5)
      name = os.path.basename(sm_path)
      print("@@ %s, average time(ms) is %.4f" % (name, avg_time*1000))
      print('FINISH')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "run onnx xla model")
    parser.add_argument("onnx", help = "onnx model path")
    parser.add_argument("-d", "--device", default="x86", choices=["gpu","x86", "arm"])
    parser.add_argument("-t", "--thread", default="multiple", choices=["multiple","single"])
    arg = parser.parse_args() 

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    tf.keras.backend.clear_session()
    tf.config.optimizer.set_jit(True) # Enable XLA


    #if arg.device=='x86':
    if arg.thread=='single':
      tf.config.threading.set_inter_op_parallelism_threads(1)
      tf.config.threading.set_intra_op_parallelism_threads(1)

    
    print(time.strftime("[localtime] %Y-%m-%d %H:%M:%S", time.localtime()) )

    tf2xla(arg.onnx) 
