import numpy as np
from .blob import Blob


box=[]

class Base(object):
    def __init__(self,input,name=''):
        def transfer_input(_input):
            if isinstance(_input,Base):
                _input=_input()
                assert isinstance(input,Blob),'The input of layer %s is not Blob, please use nn_tools.P.blob.Blob as input'%name
            return _input
        if type(input)==list:
            # if multi input
            self.input=[transfer_input(i) for i in input]
            self.input_size = np.sum([np.prod(i.shape) for i in self.input])
            self.muti_input=True
        else:
            self.input = transfer_input(input)
            self.input_size = np.prod(self.input.shape)
            self.muti_input = False
        self.name=name
        self.weight_size=0
        self.activation_size=None
        self.dot=0
        self.add=0
        self.pow=0
        self.compare=0
        self.ops=0
        self.out=None
        self.layer_info=None
        box.append(self)

    def __call__(self, *args, **kwargs):
        return self.out
    def __setattr__(self, key, value):
        if key=='out' and value!=None:
            if type(value) is list:
                self.activation_size=0
                for i in value:
                    self.activation_size+=np.prod(i.shape)
            else:
                self.activation_size=np.prod(value.shape)
        return object.__setattr__(self, key,value)
    def __getattribute__(self, item):
        if item=='ops':
            try:
                self.ops=self.pow+self.add+self.dot+self.compare
            except:
                print("CRITICAL WARNING: Layer {} ops cannot be calculated, set to 0.".format(self.name))
                self.ops=0
        return object.__getattribute__(self,item)

class Norm(Base):
    valid_tuple=('norm','batch_norm','lrn')
    def __init__(self,input,type,name=None):
        if type not in Norm.valid_tuple:
            raise NameError('the norm type:' + type + ' is not supported. ' \
                             'the valid type is: ' + str(Activation.valid_tuple))
        if name == None: name = type
        Base.__init__(self, input, name=name)
        getattr(self, type)()
        self.out = self.input.new(self)

    def norm(self):
        self.dot = self.input_size
        self.add = self.input_size

    def batch_norm(self):
        self.dot = self.input_size
        self.add = self.input_size

    def lrn(self):
        self.dot = self.input_size
        self.add = self.input_size

class Activation(Base):
    #valid tuple lists the valid activation function type
    valid_tuple=('relu','tanh','prelu')
    def __init__(self,input,type,name=None):
        if type not in Activation.valid_tuple:
            raise NameError('the activation type:'+type+' is not supported. ' \
                            'the valid type is: '+str(Activation.valid_tuple))
        if name==None:name=type
        Base.__init__(self,input,name=name)
        getattr(self,type)()
        self.out=self.input.new(self)

    def relu(self):
        self.compare=self.input_size

    def sigmoid(self):
        self.add=self.dot=self.pow=self.input_size

    def tanh(self):
        self.dot=self.input_size
        self.add=self.pow=self.input_size*2

    def prelu(self):
        self.compare=self.input_size
        self.dot=self.input_size


class Sliding(Base):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,name='sliding',ceil=False,transpose=False):
        # input is the instance of blob.Blob with shape (c,h,w) or (batch,c,h,w)
        super(Sliding,self).__init__(input,name=name)
        if self.input.dim==4:
            conv_dims=2
        elif self.input.dim==5:
            conv_dims=3
            self.input_t=self.input.t
        else:
            raise ValueError('Sliding must have a input with 2D Map(batch,c,w,h) or 3D Map(batch,c,d,w,h)')
        self.input_w = self.input.w
        self.input_h = self.input.h
        self.batch_size = self.input.batch_size
        self.in_channel = self.input.c

        if type(kernel_size) == int:
            self.kernel_size = [kernel_size] * conv_dims
        else:
            assert len(kernel_size)==conv_dims
            self.kernel_size = [i for i in kernel_size]
        if type(stride) == int:
            self.stride = [stride] * conv_dims
        else:
            self.stride = [i for i in stride]
            if len(self.stride) == 1:
                self.stride = [self.stride[0]] * conv_dims
            elif len(self.stride) == 0:
                self.stride = [1] * conv_dims
        if type(pad) == int:
            self.pad = [pad] * conv_dims
        else:
            self.pad = [i for i in pad]
            if len(self.pad) == 1:
                self.pad *= conv_dims
            elif len(self.pad) == 0:
                self.pad = [0] * conv_dims
        self.num_out = num_out
        self.layer_info ='kernel=%s,stride=%s,pad=%s'%('x'.join([str(_) for _ in self.kernel_size]),
                                                       'x'.join([str(_) for _ in self.stride]),
                                                       'x'.join([str(_) for _ in self.pad]))
        if transpose:
            self.layer_info += ',transpose'
        # calc out

        outs=[]

        for i in range(self.input.dim-2):
            if not transpose:
                if not ceil:
                    outs.append(np.floor(float(self.input[2+i] + self.pad[i] * 2 - self.kernel_size[i]) / self.stride[i]) + 1)
                else:
                    outs.append(np.ceil(float(self.input[2+i] + self.pad[i] * 2 - self.kernel_size[i]) / self.stride[i]) + 1)
            else:
                # transpose
                outs.append((self.input[2+i] - 1) * self.stride[i] - 2 * self.pad[i] + self.kernel_size[i])
        #     if not ceil:
        #         out_h = np.floor(float(self.input_w + self.pad[0] * 2 - self.kernel_size[0]) / self.stride[0]) + 1
        #         out_w = np.floor(float(self.input_h + self.pad[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1
        #         out_t = np.floor(float(self.input_t + self.pad[2] * 2 - self.kernel_size[2]) / self.stride[2]) + 1
        #     else:
        #         out_w = np.ceil(float(self.input_w + self.pad[0] * 2 - self.kernel_size[0]) / self.stride[0]) + 1
        #         out_h = np.ceil(float(self.input_h + self.pad[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1
        #         out_t = np.ceil(float(self.input_h + self.pad[1] * 2 - self.kernel_size[1]) / self.stride[1]) + 1
        # else:
        #     # transpose
        #     out_w = (self.input_w - 1) * self.stride[0] - 2 * self.pad[0] + self.kernel_size[0]
        #     out_h = (self.input_h - 1) * self.stride[1] - 2 * self.pad[1] + self.kernel_size[1]

        self.out = Blob([self.batch_size, num_out, *outs], self)

class Conv(Sliding):
    def __init__(self,input,kernel_size,num_out,stride=1,pad=0,
                 activation='relu',name='conv',ceil=False,group_size=1,transpose=False):
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,num_out,stride,pad,name=name,ceil=ceil,transpose=transpose)
        self.layer_info+=',num_out=%d'%(num_out)
        self.dot = np.prod(self.out.shape) * np.prod(self.kernel_size) * self.in_channel
        self.weight_size = np.prod(self.kernel_size) * num_out * self.in_channel
        if group_size!=1:
            self.layer_info += ',group_size=%d' % (group_size)
            self.dot /= group_size
            self.weight_size /= group_size
        self.add = self.dot
        if activation:
            Activation(self.out,activation)

class Pool(Sliding):
    def __init__(self,input,kernel_size,stride=1,pad=0,name='pool',pool_type='max',ceil=False):
        # pool_type: 0 is max, 1 is avg/ave in Caffe
        if isinstance(input,Base):
            input=input()
        Sliding.__init__(self,input,kernel_size,input.c,stride,pad,name=name,ceil=ceil)
        self.pool_type=pool_type
        self.layer_info+=',type=%s'%(pool_type)
        if pool_type in ['max',0]:
            self.compare= np.prod(self.out.shape) * (np.prod(self.kernel_size) - 1)
        elif pool_type in ['avg','ave',1]:
            self.add = np.prod(self.input.shape)
            self.dot = np.prod(self.out.shape)
        else:
            print("WARNING, NOT IMPLEMENT POOL TYPE %s PROFILING at %s, CONTINUE"%(pool_type,name))
pool=Pool

class InnerProduct(Base):
    def __init__(self,input,num_out,activation='relu',name='innerproduct'):
        if isinstance(input,Base):
            input=input()
        Base.__init__(self,input,name=name)
        self.left_dim=np.prod(input.shape[1:])
        self.num_out=num_out
        self.dot=self.num_out*self.input_size
        self.add=self.num_out*self.input_size
        self.out=Blob([input[0],self.num_out],self)
        self.weight_size = self.num_out * self.left_dim
        if activation:
            Activation(self.out,activation)
Fc=InnerProduct
fc=InnerProduct

class Permute(Base):
    def __init__(self, input,dims, name='permute'):
        super(Permute,self).__init__(input,name)
        self.out = Blob(dims,self)

class Flatten(Base):
    def __init__(self,input, name='permute'):
        super(Flatten, self).__init__(input, name)
        dim=[np.prod(input.shape)]
        self.out = Blob(dim, self)

class Eltwise(Base):
    def __init__(self,inputs,type='sum',name='eltwise'):
        super(Eltwise,self).__init__(inputs,name,)
        self.out=inputs[0].new(self)
        if type in ['sum','SUM']:
            self.add=np.prod(self.out.shape)
        elif type in ['product','PROD']:
            self.dot=np.prod(self.out.shape)
        elif type in ['max','MAX']:
            self.compare=np.prod(self.out.shape)
        else:
            raise AttributeError('the Eltwise layer type must be sum, max or product')

class Slice(Base):
    def __init__(self,input,slice_point,axis,name='slice'):
        super(Slice,self).__init__(input,name,)
        self.out=[]
        last=0
        for p in slice_point:
            print(p,list(input.shape))
            shape1=list(input.shape)
            shape1[axis] = p-last
            last=p
            self.out+=[Blob(shape1)]
        shape1 = list(input.shape)
        print(last,shape1,input.shape[axis])
        shape1[axis] = input.shape[axis] - last
        self.out += [Blob(shape1)]

class Reshape(Base):
    def __init__(self,input,shape,name='reshape'):
        super(Reshape,self).__init__(input,name)
        shape=list(shape)
        for i in range(len(shape)):
            if shape[i]==0:
                shape[i]=input.shape[i]
        self.out=Blob(shape)


class Concat(Base):
    def __init__(self,inputs,axis,name='concat'):
        super(Concat,self).__init__(inputs,name,)
        outc=0
        for input in inputs:
            outc+=input[axis]
        self.out=Blob(inputs[0].shape,self)
        self.out.shape[axis]=outc

class Scale(Base):
    def __init__(self, input, factor=None, name='scale'):
        super(Scale, self).__init__(input, name, )
        self.out = input.new(self)

        self.dot=self.input_size
        # TODO scale analysis

class Softmax(Base):
    def __init__(self, input, factor=None, name='softmax'):
        super(Softmax, self).__init__(input, name, )
        self.out = input.new(self)
        self.power=self.input_size
        self.add=self.input_size
        self.dot=self.input_size
        self.layer_info="softmax"

class Dropout(Base):
    def __init__(self,input,name='dropout'):
        if isinstance(input,Base):
            input=input()
        Base.__init__(self,input,name=name)
        self.out = input.new(self)