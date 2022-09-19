import torch, sys
import torch.nn as nn
import time
sys.path.append('.')
import os
from easydict import EasyDict as edict
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from options import stage2_opts
from utils import logger, recorders
from datasets import custom_data_loader
from models import custom_model, solver_utils, model_utils
import pdb
import train_stage2 as train_utils


from itertools import chain

args = stage2_opts.TrainOpts().parse()
log = logger.Logger(args)


epochs = 10

def parseData(sample):
    img, normal, mask = sample['img'], sample['normal'], sample['mask']
    dirs = sample['dirs']
    dir_coll = sample['dir_coll']
    img_coll = sample['img_coll']
    img, normal, mask = img.cuda(), normal.cuda(), mask.cuda()
    dirs = dirs.cuda()
    dir_coll = dir_coll.cuda()
    img_coll = img_coll.cuda()
    coarse_normal = sample['coarse_normal'].cuda()
    refine_normal = sample['refine_normal'].cuda()
    data = {'img': img, 'gd': normal, 'm': mask, 'dirs': dirs, 'dir_coll':dir_coll, 'img_coll':img_coll,'coarse_normal':coarse_normal, 'refine_normal': refine_normal}
    return data

def getInput(data):
    input_list = [data['img']]
    input_list.append(data['gd'])
    input_list.append(data['dirs'])
    input_list.append(data['m'])
    input_list.append(data['dir_coll'])
    input_list.append(data['img_coll'])
    return input_list

class loss1_criterion():
    def __init__(self):
        self.n_crit = torch.nn.CosineEmbeddingLoss()  # cos_loss
        self.n_crit = self.n_crit.cuda()

        self.img_crit = torch.nn.MSELoss()
        self.img_crit = self.img_crit.cuda()

    def forward(self, pred_normal, x, pred_fine, output, target, mask, f):
        normal_gt = x
        normal_pred = pred_normal
        coarse_pred = pred_fine
        self.loss = 0
        coarse_loss = 0
        out_loss = {}
        n_est, n_tar = normal_pred, normal_gt
        n_num = n_tar.nelement() // n_tar.shape[1]
        n_flag = n_tar.data.new().resize_(n_num).fill_(1)
        out_reshape = n_est.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        out_reshape_coarse = coarse_pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt_reshape = n_tar.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        coarse_normal_loss = self.n_crit(out_reshape_coarse, gt_reshape, n_flag)
        normal_loss = self.n_crit(out_reshape, gt_reshape, n_flag)

        self.loss += normal_loss
        coarse_loss += coarse_normal_loss
        out_loss['Normal_loss'] = self.loss.item()
        out_loss['Fine_loss'] = coarse_loss.item()

        image_loss = 0
        target = torch.split(target, 3, 1)
        img_est, img_tar = output, target
        for i in range(len(target)):
            est = torch.mul(mask, img_est[i])
            tar = torch.mul(mask, img_tar[i])
            image_loss += self.img_crit(est, tar)
        self.loss += f * image_loss
        out_loss['img_loss'] = f * image_loss.item()

        return out_loss

    def backward(self):
        self.loss.backward()


def Refine_saveCheckpoint(save_path, epoch=-1, model=None, model_name=None):
    state = {'state_dict': model.state_dict(), 'model': model_name}
    torch.save(state, os.path.join(save_path, 'Refine_{}.pth.tar'.format(epoch)))


def Recons_saveCheckpoint(save_path, epoch=-1, model=None, model_name=None):
    state = {'state_dict': model.state_dict(), 'model': model_name}
    torch.save(state, os.path.join(save_path, 'Recons_{}.pth.tar'.format(epoch)))


def buildModelStage1(args):
    print('Creating Stage1 Model model_refine')
    models = __import__('models.' + args.model_refine)
    model_file = getattr(models, args.model_refine)
    model = getattr(model_file, args.model_refine)
    model = model()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    if args.retrain_refine:
        args.log.printWrite("=> using pre-trained model_refine '{}'".format(args.retrain_refine))
        model_utils.loadCheckpoint(args.retrain_refine, model, cuda=args.cuda)

    print(model)
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print("=> Stage1 Refine Model Parameters: %d" % pp)
    return model

def buildModelStage2(args):
    print('Creating Stage1 Model model_recons')
    models = __import__('models.' + args.model_recons)
    model_file = getattr(models, args.model_recons)
    model = getattr(model_file, args.model_recons)
    model = model()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()

    print(model)
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print("=> Stage2 Reconstruction Model Parameters: %d" % pp)
    return model


def main(args):
    train_loader, val_loader = custom_data_loader.customDataloader(args)
    model_Refine = buildModelStage1(args)
    model_Recons = buildModelStage2(args)

    optimizer = torch.optim.Adam([
        {'params': model_Refine.parameters(), 'lr': 0.001},
        {'params': model_Recons.parameters(), 'lr': 0.001}
    ], betas=(0.9, 0.999))
    criterion = loss1_criterion()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[2, 4, 6, 8, 10], gamma=0.5,
                                                     last_epoch=-1)
    for epoch in range(1, epochs + 1):
        # epoch = 2, 4, 6, 8 --> 0.2, 0.4, 0.6, 0.8, 1
        if epoch == 1 or epoch == 2:
            f = 0.2
        if epoch == 3 or epoch == 4:
            f = 0.4
        if epoch == 5 or epoch == 6:
            f = 0.6
        if epoch == 7 or epoch == 8:
            f = 0.8
        if epoch == 9 or epoch == 10:
            f = 1
        print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(train_loader)))
        model_Refine.train()
        model_Recons.train()
        for i, sample in enumerate(train_loader):
            data = parseData(sample)
            input = getInput(data)
            coarse = data['coarse_normal']
            refine = data['refine_normal']
            pred_normal, recons_fused = model_Refine(input, coarse)
            pred_image  = model_Recons(input, pred_normal, recons_fused)
            optimizer.zero_grad()

            loss1 = criterion.forward(pred_normal['n'], input[1], refine, pred_image, input[0], input[3], f)
            criterion.backward()
            optimizer.step()
            iters = i + 1
            if i % 20 == 0:
                print('epoch :', epoch)
                print('迭代次数 :  %d: %d' % (iters, len(train_loader)))
                print('Images_loss :', loss1['img_loss'])
                print('Normal_loss :', loss1['Normal_loss'])
                print('Fine_loss   :', loss1['Fine_loss'])

        scheduler.step()

        if epoch % args.save_intv == 0 :
            Refine_saveCheckpoint(args.cp_dir, epoch, model_Refine, args.model_refine)
            Recons_saveCheckpoint(args.cp_dir, epoch, model_Recons, args.model_recons)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)

'''
参数设置：
--batch 8 
--collocated
--normalize
--model_refine Refine_PS_5conv
--model_recons Recons_net
--retrain_refine data/models/5conv_Refine_7.pth.tar
--in_img_num 64 
--retrain data/logdir/path/to/checkpointDirOfLCNet/checkpoint20.pth.tar 
--model_s2 PS_FCN_collocated 
--retrain_s2 data/models/PS_FCN_collocated_22.pth.tar 

nohup python -u \
Coarse_Fine_Recons.py --batch 8 --collocated --normalize --in_img_num 64 \
--model_refine Refine_PS_5conv --model_recons Recons_net \
--retrain_refine data/models/5conv_Refine_7.pth.tar \
--retrain data/logdir/path/to/checkpointDirOfLCNet/checkpoint20.pth.tar --model_s2 PS_FCN_collocated --retrain_s2 data/models/PS_FCN_collocated_22.pth.tar \
>python.log 2>&1 &
'''

