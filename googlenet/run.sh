# make sure it is python3.4, otherwise some scipy.misc functions will not work
# To train
CUDA_VISIBLE_DEVICES=0 python3 train.py -m train -dd /path/to/streetstyle27k/ -mn mark1

# To test with a saved model
CUDA_VISIBLE_DEVICES=0 python3 train.py -m test -dd /path/to/streetstyle27k/ -mn mark1

# To run inference with pretrained model use googlenet_infer.py
CUDA_VISIBLE_DEVICES=0 python3 inference.py


