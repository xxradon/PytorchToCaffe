import torch
import numpy as np

class Resize_preprocess(object):
    """Rescales the input PIL.Image to the given 'size_w,size_h'.
    """

    def __init__(self, size_w,size_h):
        self.size = (size_w,size_h)

    def __call__(self, img):
        return img.resize(self.size)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_mean_std(loader):
    # the image should be preprocessed by torch.transform.ToTensor(), so the value is in [0,1]
    sum=np.ones(3)
    cnt=0
    for datas,_ in loader:
        cnt+=len(datas)
        for data in datas:
            data=data.numpy()
            sum+=data.sum(1).sum(1)/np.prod(data.shape[1:])
    mean=sum/cnt
    error=np.ones(3)
    _mean=mean.reshape([3,1,1])
    for datas,_ in loader:
        cnt+=len(datas)
        for data in datas:
            data=data.numpy()
            error+=((data-_mean)**2).sum(1).sum(1)/np.prod(data.shape[1:])
    std=np.sqrt(error/cnt)
    return mean,std