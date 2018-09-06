import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
import pytorch_to_caffe

if __name__=='__main__':
    name='inception_v3'
    net=inception_v3(True,transform_input=False)
    net.eval()
    input=Variable(torch.ones([1,3,299,299]))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))