import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
#from torchvision.models import resnet
from model import resnet
import pytorch_to_caffe

if __name__=='__main__':
    name='resnet50'
    resnet50=resnet.resnet50(num_classes=2)
    
    #resnet18.load_state_dict(checkpoint)
    resnet50.eval()
    input=torch.ones([1,3,160,160])
    pytorch_to_caffe.trans_net(resnet50,input,name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))
