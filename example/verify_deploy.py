# 2018.09.06 by Shining 
import sys
sys.path.insert(0,'/home/shining/Projects/github-projects/caffe-project/caffe/python')
import caffe
import torchvision.transforms as transforms
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.models import resnet
import time



import cv2

#caffe load formate
def load_image_caffe(imgfile):
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, args.height, args.width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, args.height, args.width)
    return image

def load_image_pytorch(imgfile):
    
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # # transforms.ToTensor()
    # transform1 = transforms.Compose([
    #     transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    #     normalize
    #     ]
    # )
    # ##numpy.ndarray
    # img = cv2.imread(imgfile)# 读取图像
    # img = cv2.resize(img,(224,244))
    # img1 = transform1(img) # 归一化到 [0.0,1.0]
    # print("img1 = ",img1)

    img = np.ones([1,3,args.height, args.width])
    # 转化为numpy.ndarray并显示
    return img



def forward_pytorch(weightfile, image):
    net=resnet.resnet18()
    checkpoint = torch.load(weightfile)
    net.load_state_dict(checkpoint)
    if args.cuda:
        net.cuda()
    print(net)
    net.eval()
    image = torch.from_numpy(image)
    if args.cuda:
        image = Variable(image.cuda())
    else:
        image = Variable(image)
    t0 = time.time()
    blobs = net.forward(image)
    #print(blobs.data.numpy().flatten())
    t1 = time.time()
    return t1-t0, blobs, net.parameters()

# Reference from:
def forward_caffe(protofile, weightfile, image):
    if args.cuda:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(protofile, weightfile, caffe.TEST)
    net.blobs['blob1'].reshape(1, 3, args.height, args.width)
    net.blobs['blob1'].data[...] = image
    t0 = time.time()
    output = net.forward()
    t1 = time.time()
    return t1-t0, net.blobs, net.params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('--protofile', default='/home/shining/Projects/github-projects/pytorch-project/nn_tools/MGN.prototxt', type=str)
    parser.add_argument('--weightfile', default='/home/shining/Projects/github-projects/pytorch-project/nn_tools/MGN.caffemodel', type=str)
    parser.add_argument('--model', default="/home/shining/Projects/github-projects/pytorch-project/nn_tools/model_100.pt", type=str)
    parser.add_argument('--imgfile', default='/home/shining/Projects/github-projects/pytorch-project/nn_tools/001763.jpg', type=str)
    parser.add_argument('--height', default=384, type=int)
    parser.add_argument('--width', default=128, type=int)
    parser.add_argument('--meanB', default=104, type=float)
    parser.add_argument('--meanG', default=117, type=float)
    parser.add_argument('--meanR', default=123, type=float)
    parser.add_argument('--scale', default=255, type=float)
    parser.add_argument('--synset_words', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)
    
    
    protofile = args.protofile
    weightfile = args.weightfile
    imgfile = args.imgfile

    image = load_image_pytorch(imgfile)
    time_pytorch, pytorch_blobs, pytorch_models,out_Tensor_pytorch = forward_pytorch(args.model, image)
    time_caffe, caffe_blobs, caffe_params,out_Tensor_caffe = forward_caffe(protofile, weightfile, image)

    print('pytorch forward time %d', time_pytorch)
    print('caffe forward time %d', time_caffe)
    
    print('------------ Output Difference ------------')
    blob_name = "cat_blob1"
    if args.cuda:
        pytorch_data = pytorch_blobs.data.cpu().numpy().flatten()
    else:
        pytorch_data = pytorch_blobs.data.numpy().flatten()
    caffe_data = caffe_blobs[blob_name].data[0][...].flatten()
    diff = abs(pytorch_data - caffe_data).sum()
    print('%-30s pytorch_shape: %-20s caffe_shape: %-20s output_diff: %f' % (blob_name, pytorch_data.shape, caffe_data.shape, diff/pytorch_data.size))
