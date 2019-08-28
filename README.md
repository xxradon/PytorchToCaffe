# The code is mainly come from [nn_tools](https://github.com/hahnyuan/nn_tools).Thanks for hahnyuan's contribution.
# Neural Network Tools: Converter and Analyser

 Providing a tool for  neural network frameworks for pytorch and caffe.
 
 The nn_tools is released under the MIT License (refer to the LICENSE file for details).

### features

1. Converting a pytorch model to caffe model.
2. Some convenient tools of manipulate caffemodel and prototxt quickly(like get or set weights of layers).
3. Support pytorch version >= 0.2.(Have tested on 0.3,0.3.1, 0.4, 0.4.1 ,1.0,1.1,1.2)
4. Analysing a model, get the operations number(ops) in every layers.

### requirements

- Python2.7 or Python3.x
- Each functions in this tools requires corresponding neural network python package (pytorch and so on).

# Analyser

The analyser can analyse all the model layers' [input_size, output_size, multiplication ops, addition ops, 
comparation ops, tot ops, weight size and so on] given a input tensor size, which is convenint for model deploy analyse.

## Caffe
Before you analyse your network, [Netscope](http://ethereon.github.io/netscope/#/editor)
is recommended to visiualize your network.

Command：`python caffe_analyser.py [-h] prototxt outdir shape`
- The prototxt is the path of the prototxt file.
- The outdir is path to save the csv file.
- The shape is the input shape of the network(split by comma `,`), in caffe image shape should be: 
batch_size, channel, image_height, image_width.

For example `python caffe_analyser.py resnet_18_deploy.prototxt analys_result.csv 1,3,224,224`

## Pytorch
Supporting analyse the inheritors of torch.nn.Moudule class.

Command：`pytorch_analyser.py [-h] [--out OUT] [--class_args ARGS] path name shape`
- The path is the python file path which contaning your class.
- The name is the class name or instance name in your python file.
- The shape is the input shape of the network(split by comma `,`), in pytorch image shape should be:
batch_size, channel, image_height, image_width.
- The out (optinal) is path to save the csv file, default is '/tmp/pytorch_analyse.csv'.
- The class_args (optional) is the args to init the class in python file, default is empty.

For example `python pytorch_analyser.py example/resnet_pytorch_analysis_example.py resnet18 1,3,224,224`



# Converter

## Pytorch to Caffe

The new version of pytorch_to_caffe supporting the newest version(from 0.2.0 to 1.2.0) of pytorch.
NOTICE: The transfer output will be somewhat different with the original model, caused by implementation difference.

- Supporting layers types: 
conv2d  ->  Convolution, 
_conv_transpose2d ->  Deconvolution, 
_linear -> InnerProduct, 
_split  -> Slice, 
max_pool2d,_avg_pool2d   -> Pooling,
_max    -> Eltwise, 
_cat    -> Concat,
dropout -> Dropout,
 relu   -> ReLU, 
 prelu  -> PReLU, 
 _leaky_relu -> ReLU,
 _tanh  -> TanH,   
 threshold(only value=0) -> Threshold,ReLU,
 softmax -> Softmax, 
 batch_norm -> BatchNorm,Scale, 
 instance_norm -> BatchNorm,Scale,
 _interpolate  ->  Upsample
 
- Supporting operations: torch.split, torch.max, torch.cat ,torch.sigmoid
- Supporting tensor Variable operations: var.view, + (add), += (iadd), -(sub), -=(isub)
 \* (mul) *= (imul)

Need to be added for caffe in the future:
- Normalize,DepthwiseConv

The supported above can transfer many kinds of nets, 
such as AlexNet(tested), VGG(tested), ResNet(fixed the bug in origin repo which mainly caused by ReLu layer function.), Inception_V3(tested).

The supported layers concluded the most popular layers and operations.
 The other layer types will be added soon, you can ask me to add them in issues.

Example: please see file `example/alexnet_pytorch_to_caffe.py`. Just Run `python3 example/alexnet_pytorch_to_caffe.py`.

Attention:
the main difference from convert model is the BN layer,you should pay more attention to the BN parameters like  momentum=0.1, eps=1e-5.

# Deploy verify(Very Important)
After Converter,we should use verify_deploy.py to verify the output of pytorch model and the convertted caffe model.
If you want to verify the outputs of caffe and pytorch,you should make caffe and pytorch install in the same environment,anaconda is recommended.
using following script,we can install caffe-gpu(master branch). 
```angular2html
conda install caffe-gpu pytorch cudatoolkit=9.0 -c pytorch 

```
other way,we can use docker,and in https://github.com/ufoym/deepo,for cuda9
```
docker pull ufoym/deepo:all-py36-cu90
```
for cuda10
```
docker pull ufoym/deepo:all-py36-cu100
```

please see file `example/verify_deploy.py`,it can verify the output of pytorch model and the convertted caffe model in the same input.


# Some common functions

## funcs.py

- **get_iou(box_a, box_b)** intersection over union of two boxes
- **nms(bboxs,scores,thresh)** Non-maximum suppression
- **Logger** print some str to a file and stdout with H M S

