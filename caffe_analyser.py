# coding=utf-8
from __future__ import absolute_import
import argparse
from analysis.CaffeA import *
from Caffe import caffe_net
from analysis.utils import save_csv

"""
Before you analyse your network, [Netscope](http://ethereon.github.io/netscope/#/editor)
is recommended to visiualize your network.

Command：`python caffe_analyser.py [-h] prototxt outdir shape`
- The prototxt is the path of the prototxt file.
- The outdir is path to save the csv file.
- The shape is the input shape of the network(split by comma `,`), image shape should be: channel, batch_size, image_height, image_width.

For example `python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,3,224,224`
"""

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('prototxt',help='path of the prototxt file',type=str)
    parser.add_argument('outdir',help='path to save the csv file',type=str)
    parser.add_argument('shape',help='input shape of the network(split by comma `,`), image shape should be: batch,c,h,w',type=str)
    args=parser.parse_args()
    shape=[int(i) for i in args.shape.split(',')]
    net=caffe_net.Prototxt(args.prototxt)
    blob_dict, layers=profiling(net, Blob(shape))
    save_csv(layers,args.outdir)