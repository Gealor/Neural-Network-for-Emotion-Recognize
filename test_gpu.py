import tensorflow as tf
import torch

print("[Tensorflow] Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("[PyTorch] GPU Available: ", torch.cuda.is_available() == True)