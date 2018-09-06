from torch.autograd import Variable
from .utils import AverageMeter

def eval_classification_net(net,testloader,use_cuda=True):
    # evaluation a classification net
    # the testloader should return (input, labels(not one-hot version))
    # return an Tensor with shape [1] for accuracy(%)
    net.train(False)
    top1 = AverageMeter()
    for inputs, targets in testloader:
        if use_cuda:
            inputs=inputs.cuda()
            targets=targets.cuda()
        outputs = net(Variable(inputs))
        res1,=compute_accuracy(outputs, targets, topk=(1,))
        top1.update(res1)
    return top1.avg[0]

def eval_classification_net_topk(net,testloader,use_cuda=True,topk=(1,)):
    # evaluation a classification net
    # the testloader should return (input, labels(not one-hot version))
    # return an Tensor with shape [1] for accuracy(%)
    net.train(False)
    tops=[]
    for i in topk:
        tops.append(AverageMeter())
    for inputs, targets in testloader:
        if use_cuda:
            inputs=inputs.cuda()
            targets=targets.cuda()
        outputs = net(Variable(inputs))
        res=compute_accuracy(outputs, targets, topk=topk)
        for i,_ in enumerate(topk):
            tops[i].update(res[i])
    return [i.avg[0] for i in tops]

def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if isinstance(output,Variable):
        output=output.data
    if isinstance(target,Variable):
        target=target.data
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res