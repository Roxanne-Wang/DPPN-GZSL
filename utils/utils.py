import numpy as np
import torch.nn as nn
import scipy.sparse as sp
#from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from PIL import ImageFilter
import random
import torchvision.transforms as transforms
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                 

@torch.no_grad()
def all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return tensors_gather
    
    
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x,axis=1,keepdims=True)
    return softmax_x 

def compute_class_accuracy_total( true_label, predict_label, classes):
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))
    return np.mean(acc_per_class)


def compute_domain_accuracy(predict_label, domain):
    num = predict_label.shape[0]
    n = 0
    for i in predict_label:
        if i in domain:
            n +=1
            
    return float(n)/num

def entropy(probs): 
    """ Computes entropy. """ 
    max_score = np.max(probs,axis=1)   
    return -max_score * np.log(max_score)

            
def ood_opt(v_prob,a_prob,gt, split_num, seen_c,unseen_c):
    v_max = np.max(v_prob,axis=1)
    H_v = entropy(v_prob)   
    v_pre = np.argmax(v_prob,axis=1)
    
    a_max = np.max(v_prob,axis=1)
    H_a = entropy(a_prob)
    a_pre = np.argmax(a_prob,axis=1) 
        
    opt_S = 0
    opt_U = 0
    opt_H = 0
    opt_Ds = 0
    opt_Du = 0
    opt_tau = 0
    
    for step in range(9):
        base = 0.1*step+0.1
        tau = -base* np.log(base)
        pre = v_pre
        for idx,class_i in enumerate(pre):
            if(v_max[idx]-base<0):
                pre[idx] = a_pre[idx]
                
        pre_s = pre[:split_num];pre_t = pre[split_num:]
        gt_s = gt[:split_num];gt_t = gt[split_num:]
        S = compute_class_accuracy_total(gt_s, pre_s,seen_c)
        U = compute_class_accuracy_total(gt_t, pre_t,unseen_c)
        Ds = compute_domain_accuracy(pre_s,seen_c)
        Du = compute_domain_accuracy(pre_t,unseen_c)
        H = 2*S*U/(S+U) 
        
        #print('S: {:.4f} U {:.4f} H {:.4f} Ds {:.4f} Du_{:.4f} tau {:.4f}'.format(S, U,H,Ds,Du,base))
         
        if H>opt_H:
             opt_S = S
             opt_U = U
             opt_H = H
             opt_Ds = Ds
             opt_Du = Du
             opt_tau = tau
            
    return opt_H,opt_S,opt_U,opt_Ds,opt_Du,opt_tau

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
def swap(img, crop):
    def crop_image(image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    widthcut, highcut = img.size
    img = img.crop((10, 10, widthcut-10, highcut-10))
    images = crop_image(img, crop)
    pro = 5
    if pro >= 5:          
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 1
        RAN = 2
        for i in range(crop[1] * crop[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
            if count_x == crop[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                random.shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
        random_im = []
        for line in tmpy:
            random_im.extend(line)
        
        # random.shuffle(images)
        width, high = img.size
        iw = int(width / crop[0])
        ih = int(high / crop[1])
        toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
        x = 0
        y = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            x += 1
            if x == crop[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    return toImage

class Randomswap(object):
    def __init__(self, size):
        self.size = size
        self.size = (int(size), int(size))

    def __call__(self, img):
        return swap(img, self.size)
        
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
    
    
    
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, lr, epoch_decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // epoch_decay))
    if optimizer != None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
