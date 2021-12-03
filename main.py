#from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import sys
import torchvision.transforms as transforms
import datasets
import models
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
''' opt for data '''
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--data-name', default='cub', type=str, help='dataset')     
# multi-gpu
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')  
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--seed1', default=None, type=int,
                    help='seed1 for initializing training. ')
parser.add_argument('--seed2', default=None, type=int,
                    help='seed2 for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
''' opt for optimizer'''
parser.add_argument('--opti_type', default='adam', help='optimizer type default:adam')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', default=30, type=int,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
parser.add_argument('--save_H_OPT', action='store_true', help='save H_opt best model')
''' opt for model '''
parser.add_argument('--backbone', default='resnet101', help='')
parser.add_argument('--model', default='dpn', help='')
parser.add_argument('--n-dec', default=3, type=int, help='')
parser.add_argument('--vis-emb-dim', default=512, type=int, help='')
parser.add_argument('--att-emb-dim', default=8, type=int, help='')
parser.add_argument('--w_L_cls', default=1.0, type=float, help='')
parser.add_argument('--w_L_zsl', default=1.0, type=float, help='')
''' opt for others '''
parser.add_argument('--save-path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('--resnet_pretrain', default='/data/cq14/CUB/models/resnet101-5d3b4d8f.pth',
                    help='resnet pretrained model path')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--eval_only', action='store_true', help='only eval without training')


def main():
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    ''' random seed '''
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)
    
def main_worker(ngpus_per_node, args):
    best_prec1 = 0
    ''' multi-gpu '''
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))        
    ''' logger '''
    if args.is_fix:
        phase='fix'
    else:
        phase='ft'
    args.logger = setup_logger(output=args.save_path, phase=phase)
    args.logger.info(args)  
    random.seed(args.seed1)
    np.random.seed(args.seed2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
        
    ''' Data Load '''
    # data load info
    data_info = h5py.File(os.path.join('./data',args.data_name,'data_info.h5'), 'r')
    nc = data_info['all_att'][...].shape[0]
    sf_size = data_info['all_att'][...].shape[1]
    semantic_data = {'seen_class':data_info['seen_class'][...],
                     'unseen_class': data_info['unseen_class'][...],
                     'all_class':np.arange(nc),
                     'all_att': data_info['all_att'][...]}
    #load semantic data
    args.num_classes = nc
    args.sf_size = sf_size
    args.sf = semantic_data['all_att']
    
    traindir = os.path.join('./data',args.data_name,'train.list')
    valdir1 = os.path.join('./data',args.data_name,'test_seen.list')
    valdir2 = os.path.join('./data',args.data_name,'test_unseen.list')

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomApply([Randomswap(3)], p=0.2),
        transforms.ToTensor(),
        normalize
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(480),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        normalize,
    ])  
    train_dataset = datasets.ImageFolder(args.data,traindir,train_transforms)
    val_dataset1 = datasets.ImageFolder(args.data,valdir1, val_transforms)
    val_dataset2 = datasets.ImageFolder(args.data,valdir2, val_transforms)
    

    train_sampler = None
    val_sampler1 = None
    val_sampler2 = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader1 = torch.utils.data.DataLoader(
        val_dataset1, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler1)
        
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data,valdir2, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler2)
    ''' model building '''
    if args.model=='dpn':
        model = models.DPN(pretrained=True,args=args)
    elif args.model=='dpn_ood':
        model = models.DPN_ood(pretrained=True,args=args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    criterions = [nn.CrossEntropyLoss().cuda(),
                 nn.MSELoss().cuda(),
                 ]

    ''' optimizer '''
    if 'ood' in args.model:
        ood_params = [v for k, v in model.named_parameters() if 'ood' in k]
        zsr_params = [v for k, v in model.named_parameters() if 'ood' not in k]
        params = model.parameters()
        od_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, ood_params),
                             0.001, momentum=args.momentum, weight_decay=args.weight_decay)
        zs_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, zsr_params), args.lr,
                                        betas=(0.5,0.999),weight_decay=args.weight_decay)
    else:
        od_optimizer = None
        zs_optimizer = None
    if args.opti_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(),args.lr, betas=(0.5,0.999),weight_decay=args.weight_decay)
    elif args.opti_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            args.logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            args.logger.info("=> no checkpoint found at '{}'".format(args.resume))   
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.epoch_decay)
        if 'ood' in args.model:
            adjust_learning_rate(od_optimizer, epoch, 0.001, args.epoch_decay)
            adjust_learning_rate(zs_optimizer, epoch, args.lr, args.epoch_decay)        
        if args.eval_only:      
            prec1 = validate(val_loader1, val_loader2, semantic_data, model, args)
            break
        
        # train for one epoch
        train(train_loader,semantic_data, model, criterions, od_optimizer, zs_optimizer, optimizer, epoch, args)       
        # evaluate on validation set
        prec1 = validate(val_loader1, val_loader2, semantic_data, model, args)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model
        if not args.is_fix or args.save_H_OPT:
            save_path = os.path.join(args.save_path,args.model+('_{:.4f}.model').format(best_prec1))
        else:
            save_path = os.path.join(args.save_path,'fix.model')
        if is_best:          
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },filename=save_path)
            args.logger.info('save model!')


def train(train_loader, semantic_data, model, criterions, od_optimizer, zs_optimizer, optimizer, epoch, args):   
    Losses = []
    Loss_s2v = AverageMeter('L_zsl', ':.4f')
    Loss_cls = AverageMeter('L_cls', ':.4f')
    Losses.append(Loss_s2v)
    Losses.append(Loss_cls)    
    if 'ood' in args.model:
        Loss_ood = AverageMeter('L_ood', ':.4f')
        Losses.append(Loss_ood)
    progress = ProgressMeter(len(train_loader),Losses, prefix="Epoch: [{}]".format(epoch))         
    # switch to train mode
    model.train()   
    if(args.is_fix):
        freeze_bn(model) 

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)#target:class number of every input
        # compute output
        logit_zsl,logit_cls,logit_ood = model(input)
        # zsl loss
        L_s2v = 0
        for logit in logit_zsl:#
            idx = torch.arange(logit.size(0)).long()
            L_s2v += (1-logit[idx,target]).mean()/len(logit_zsl)
        Loss_s2v.update(L_s2v)
        # cls loss
        L_cls = 0
        for logit in logit_cls:
            L_cls += criterions[0](logit,target)/len(logit_cls)
        Loss_cls.update(L_cls)
        # update
        total_loss = args.w_L_zsl * L_s2v + args.w_L_cls * L_cls
        if 'ood' in args.model:
            # ood loss
            L_ood = criterions[0](logit_ood,target)
            Loss_ood.update(L_ood)
        if 'ood' in args.model and args.is_fix:
            od_optimizer.zero_grad()
            L_ood.backward()
            od_optimizer.step()        
            zs_optimizer.zero_grad()
            total_loss.backward()
            zs_optimizer.step()
        else:
            if 'ood' in args.model:
                total_loss = total_loss + L_ood
            optimizer.zero_grad()            
            total_loss.backward()
            optimizer.step()        
        if i % args.print_freq == 0:
            progress.display(i,args.logger)
        
def validate(val_loader1, val_loader2, semantic_data, model, args):

    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_c = semantic_data['all_class']
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader1):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)  
            # inference
            logit_zsl, logit_cls, logit_ood  = model(input)
            logit_zsl = logit_zsl[0]      
            if 'ood' in args.model:
                logit_cls = logit_ood
            else:
                logit_cls = logit_cls[0]                       
            logit_zsl = F.softmax(logit_zsl,dim=1).cpu().numpy()
            logit_cls = F.softmax(logit_cls,dim=1).cpu().numpy()
            logit_s = logit_zsl.copy();logit_s[:,unseen_c]=-1
            logit_t = logit_zsl.copy();logit_t[:,seen_c]=-1           
            # evaluation
            if(i==0):
                gt_s = target.cpu().numpy()
                ood_logit_s = logit_cls
                zsl_logit_s = logit_zsl
                zsl_logit_sA = logit_s
                zsl_logit_sS = logit_s
                zsl_logit_sT = logit_t
            else:
                gt_s = np.hstack([gt_s,target.cpu().numpy()])
                ood_logit_s = np.vstack([ood_logit_s,logit_cls])
                zsl_logit_s = np.vstack([zsl_logit_s,logit_zsl])
                zsl_logit_sS = np.vstack([zsl_logit_sS,logit_s])
                zsl_logit_sT = np.vstack([zsl_logit_sT,logit_t])
        for i, (input, target) in enumerate(val_loader2):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            # inference
            logit_zsl, logit_cls, logit_ood  = model(input)
            logit_zsl = logit_zsl[0]            
            if 'ood' in args.model:
                logit_cls = logit_ood
            else:
                logit_cls = logit_cls[0]        
            logit_zsl = F.softmax(logit_zsl,dim=1).cpu().numpy()
            logit_cls = F.softmax(logit_cls,dim=1).cpu().numpy()
            logit_s = logit_zsl.copy();logit_s[:,unseen_c]=-1
            logit_t = logit_zsl.copy();logit_t[:,seen_c]=-1               
            # evaluation
            if(i==0):
                gt_t = target.cpu().numpy()
                ood_logit_t = logit_cls
                zsl_logit_t = logit_zsl
                zsl_logit_tT = logit_t
            else:
                gt_t = np.hstack([gt_t,target.cpu().numpy()])
                ood_logit_t = np.vstack([ood_logit_t,logit_cls])
                zsl_logit_t = np.vstack([zsl_logit_t,logit_zsl])
                zsl_logit_tT = np.vstack([zsl_logit_tT,logit_t])     
        ood_logit = np.vstack([ood_logit_s,ood_logit_t])
        zsl_logit = np.vstack([zsl_logit_s,zsl_logit_t])
        gt = np.hstack([gt_s,gt_t])        
        SS = compute_class_accuracy_total(gt_s, np.argmax(zsl_logit_sS,axis=1),seen_c)
        UU = compute_class_accuracy_total(gt_t, np.argmax(zsl_logit_tT,axis=1),unseen_c)
        CLS = compute_class_accuracy_total(gt_s, np.argmax(ood_logit_s,axis=1),seen_c)
        H1 = 0.0
        S1 = 0.0
        U1 = 0.0
        a = 0.958
        b = 1.0
        c = 0.001
        for th_pp in np.arange(a,b,c):
            zsl_logit_s1 = zsl_logit_s.copy()
            zsl_logit_t1 = zsl_logit_t.copy()
            zsl_logit_s1[:,seen_c] = zsl_logit_s1[:,seen_c] * th_pp
            zsl_logit_t1[:,seen_c] = zsl_logit_t1[:,seen_c] * th_pp
            ST = compute_class_accuracy_total(gt_s, np.argmax(zsl_logit_s1,axis=1),seen_c)
            UT = compute_class_accuracy_total(gt_t, np.argmax(zsl_logit_t1,axis=1),unseen_c)
            H = 2*ST*UT/(ST+UT)          
            if H > H1:
                H1 = H   
                S1 = ST
                U1 = UT
        if 'ood' in args.model:
            zsl_logit[:,seen_c]=-1
            opt_H,opt_S,opt_U,opt_Ds,opt_Du,opt_tau = ood_opt(ood_logit, zsl_logit, gt, gt_s.shape[0], seen_c,unseen_c)
            if opt_H > H1:
                H1 = opt_H
                S1 = opt_S
                U1 = opt_U
    args.logger.info('Evaluation results: H: {:.4f}  S: {:.4f}  U {:.4f}'.format(H1, S1, U1)) 
    if args.save_H_OPT:
        return H1
    if args.is_fix:
        return H
    else:
        return H1

if __name__ == '__main__':
    main()
