import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch
from .eval import eval_classification_net
import os
import time
from .utils import AverageMeter
from .eval import compute_accuracy
from ..funcs import Logger

def train_classification_net(net,trainloader,testloader=None,save_path='/tmp/pytorch_train_tmp.pth',base_lr=0.01,
                             num_epoch=5,use_cuda=True,optimizer=None,lr_change=None,
                             print_iter=500,save_tmp_epoch=10,log=True,criterion=None):
    # train a classification net
    # the trainloader should return (input, labels(not one-hot version))
    # the criterion is CrossEntropyLoss by default
    logger=Logger(log and save_path+'.log' or None)
    if optimizer==None:
        optimizer=optim.Adam(net.parameters(),lr=base_lr)
    if criterion==None:
        criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion=criterion.cuda()
        net=net.cuda()

    batch_time=AverageMeter()
    load_time=AverageMeter()
    top1=AverageMeter()
    end_time=time.time()
    for epoch in range(num_epoch):
        running_loss = AverageMeter()
        for iter_idx, (inputs,labels) in enumerate(trainloader,0):
            load_time.update(time.time()-end_time)
            # TODO learning rate change function
            if use_cuda:
                inputs=inputs.cuda(async=True)
                labels=labels.cuda(async=True)
            inputs_var,labels_var=Variable(inputs),Variable(labels)

            outputs=net(inputs_var)
            loss=criterion(outputs,labels_var)
            prec1, = compute_accuracy(outputs, labels, topk=(1, ))
            top1.update(prec1[0])
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.data[0])

            batch_time.update(time.time()-end_time)
            end_time = time.time()

            if iter_idx%print_iter==0:
                logger("Epoch-Iter [%d/%d][%d/%d] Time_tot/load [%f][%f] loss [%f] Prec@1 [%f]"%(
                      epoch+1,num_epoch,iter_idx,len(trainloader),batch_time.avg,load_time.avg,running_loss.avg,top1.avg))
                running_loss.reset()
                load_time.reset()
                batch_time.reset()
                top1.reset()
        if testloader:
            net.train(False)
            acc=eval_classification_net(net,testloader)
            net.train()
            logger('-- Validate at Epoch [%d] Prec@1 [%f]'%(epoch+1, acc))
        if epoch%save_tmp_epoch==0:
            torch.save(net.state_dict(), save_path+'.tmp')
            print("%s %s saved" % (time.strftime('%H:%M:%S'), save_path+'.tmp'))
    if save_path:
        torch.save(net.state_dict(), save_path)
        os.system('rm %s'%save_path+'.tmp')
