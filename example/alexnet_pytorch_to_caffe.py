import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models.alexnet import alexnet
import pytorch_to_caffe

if __name__=='__main__':
    name='alexnet'
    net=alexnet(True)
    input=Variable(torch.ones([1,3,226,226]))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))