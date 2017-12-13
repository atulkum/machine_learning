import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet50_pretrained

best_prec = 0
num_classes = 5270
resume = '../model/ResNetcheckpoint831976.pth.tar'
data = '/data/output'
batch_size = 66
evaluate = True
epochs = 5
workers = 4
print_freq = 1000

start_lr = 0.01
start_epoch = 0

def main():
    global best_prec, start_epoch
    # create model
    model = resnet50_pretrained.get_resnet50(num_classes)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
        
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valdir = os.path.join(data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(160),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    if evaluate:
        validate(val_loader, model, criterion)
        return

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes))
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1= accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        predictions = output.max(1)[1].type_as(target)

        predictions_np = predictions.data.cpu().numpy()
        target_np = target.cpu().numpy()
        for ii in range(target.size(0)):
            x = int(target_np[ii])
            y = int(predictions_np[ii])
            conf_matrix[x, y] += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))
        del output
        del loss
        del predictions
    
    np.save('conf_matrix.npy', conf_matrix)

    print(' * Prec {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

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


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    predictions = output.max(1)[1].type_as(target)
    correct = predictions.eq(target)
    correct = correct.sum()
    return correct * 1.0 / batch_size


if __name__ == '__main__':
    main()
