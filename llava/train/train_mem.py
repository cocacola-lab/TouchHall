import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from llava.train.train import train
from transformers import set_seed

if __name__ == "__main__":
    set_seed(42)
    train(attn_implementation="flash_attention_2")
