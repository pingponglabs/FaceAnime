#!/usr/bin/env python3
# coding: utf-8
import os.path as osp
from pathlib import Path
import numpy as np
import argparse,logging,time,os,datetime
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

import torch, os, pdb
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from ddfa_utils_zyt import DDFADataset, ToTensorGjz, NormalizeGjz
from ddfa_utils_zyt import str2bool, AverageMeter
from io_utils import mkdir
import matplotlib.pyplot as plt
from _4chls_utils_18LmsAid import *
from common.models.analogy_2 import ResUNetG, NetD
from common.utils import (AverageMeter, init_weights, loss_norm_l1, loss_l1, cat_fivelet)
from common.logger import (Logger, savefig)
from common.progress.bar import Bar
from common.io import save_checkpoint, torch_to_pil_image, dump_gif
from io_utils import _load, _numpy_to_cuda, _numpy_to_tensor
from triD_downSmp_zyt import _3D_downSmpl, mask_fusion
from skimage.io import imread, imsave
from margin.ArcMarginProduct import ArcMarginProduct
from backbone.cbam import CBAMResNet

from multiprocessing.pool import ThreadPool
from multiprocessing import Pool


# global (configuration)
cudnn.benchmark = True
device = torch.device("cuda")
net_path = './arcModels/Iter_1260000_net.ckpt'
net_r = CBAMResNet(50, feature_dim=512, mode='ir').to(device)
net_r.load_state_dict(torch.load(net_path)['net_state_dict'])
net_r = net_r.cuda()

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class SuperIdentityLoss(nn.Module):
    def __init__(self):
        super(SuperIdentityLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):   # X,Y are feats
        diff = torch.add(L2Norm()(X), -L2Norm()(Y))
        error = diff * diff + self.eps
        loss = torch.mean(error)     # wrong here!
        return loss

criterion_id = SuperIdentityLoss()  # for identity loss

#learning_rate = 0.00001
betas = (0.0, 0.9)
img_size=128 # zyt add
batch_size=4*2
epochs=20
workers=8
h_dim = 128
#resume = './save_files/phase1_withMask_128_sample/models/2019-07-02T20:34:02.050851/checkpoint_0012.pth.tar'
resume = './save_files/phase1_withMask_128_samTexGuided/models_idloss_v1/2019-11-05_bytedance/checkpoint_0019.pth.tar'
#resume = False

snapshot = 1
freq_disp = 100

filelists_train = "./ijbc_train_120x120_max80_refine/train_ijbc_list.txt"
#zyt add
root="./ijbc_train_120x120_max80_refine"
root_ver = "./ijbc_vertex_mat"
#depth_root = "./300vw_depth_images"

output_dir = './save_files/phase1_withMask_128_samTexGuided/exmps_sam_idloss_v1'
checkpoint_dir = './save_files/phase1_withMask_128_samTexGuided/models_idloss_v1'

filelists_val = "./ijbc_train_120x120_max80_refine/val_ijbc_list.txt"
# filelists_val_lms = "./300vw_train_120x120/testData_2dPts_frmFAN.txt"

model_gen = ResUNetG(img_size, h_dim, img_dim=3, norm_dim=3)
model_dis = NetD(img_size, input_dim=6)

model_gen = torch.nn.DataParallel(model_gen).to(device)
model_dis = torch.nn.DataParallel(model_dis).to(device)

model_gen.apply(init_weights)
model_dis.apply(init_weights)
#optim_gen = optim.Adam(model_gen.parameters(), lr=learning_rate, betas=betas)
#optim_dis = optim.Adam(model_dis.parameters(), lr=learning_rate, betas=betas)

# pool = ThreadPool(processes=16)
pool = Pool(processes=16)

def _3D_downSmpl_multiprocessing(img_size, root, root_ver, exmps_src_name, exmps_dst_name):
    assert len(exmps_src_name) == len(exmps_dst_name)
    multi_batch = []
    for idx in range(len(exmps_src_name)):
        multi_batch.append([img_size, root, root_ver, exmps_src_name[idx], exmps_dst_name[idx]])
    multi_output = pool.starmap_async(_3D_downSmpl, multi_batch)
    return multi_output.get()

def train(val_loader, train_loader, model_gen, model_dis, optim_gen, optim_dis, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    gen_losses = AverageMeter()
    dis_losses = AverageMeter()
    id_losses = AverageMeter()

    # switch to train mode
    torch.set_grad_enabled(True)
    model_gen.train()
    model_dis.train()
    end = time.time()

    #dep_image_loader = DepthImageLoader(depth_root) # zyt add

    for i, (src_img, src_name, dst_img, dst_name) in enumerate(val_loader):
        #print(i)
        
        if(i==0):
            # print('--------')
            # print(src_name)
            #pdb.set_trace()
            start = time.time()
            
            xxx = _3D_downSmpl_multiprocessing(img_size, root, root_ver, src_name, dst_name)
            # xxx = [_3D_downSmpl(img_size, root, root_ver, src, dst) for src, dst in zip(src_name, dst_name)]
            print(time.time()-start)
            vals_texMask_src = [iitem[0] for iitem in xxx]
            vals_texMask_dst = [iitem[1] for iitem in xxx]

            vals_src_heatmaps = _numpy_to_tensor(np.array(vals_texMask_src))
            vals_src_heatmaps = vals_src_heatmaps.permute(0, 3, 1, 2)
            vals_dst_heatmaps = _numpy_to_tensor(np.array(vals_texMask_dst))
            vals_dst_heatmaps = vals_dst_heatmaps.permute(0, 3, 1, 2)

            vals_x_src, vals_x_dst, vals_n_src, vals_n_dst = src_img.to(device), dst_img.to(device), \
                                                                 vals_src_heatmaps.to(device), vals_dst_heatmaps.to(device)            
        
    # print(fuck)
    bar = Bar('Train', max=len(train_loader))
    for i, (src_img, src_name, dst_img, dst_name) in enumerate(train_loader):
        
        #print(i)
        
        if(i==0):
            exmps_src_img = src_img
            exmps_src_name = src_name
            exmps_dst_img = dst_img
            exmps_dst_name = dst_name
            xxx = _3D_downSmpl_multiprocessing(img_size, root, root_ver, exmps_src_name, exmps_dst_name)
            # xxx = [_3D_downSmpl(img_size, root, root_ver, src, dst) for src, dst in zip(exmps_src_name, exmps_dst_name)]
            exmps_texMask_src = [iitem[0] for iitem in xxx]
            exmps_texMask_dst = [iitem[1] for iitem in xxx]

            exmps_src_heatmaps = _numpy_to_tensor(np.array(exmps_texMask_src))
            exmps_src_heatmaps = exmps_src_heatmaps.permute(0, 3, 1, 2)
            exmps_dst_heatmaps = _numpy_to_tensor(np.array(exmps_texMask_dst))
            exmps_dst_heatmaps = exmps_dst_heatmaps.permute(0, 3, 1, 2)

            exmps_x_src, exmps_x_dst, exmps_n_src, exmps_n_dst = exmps_src_img.to(device), exmps_dst_img.to(device), \
                                                                 exmps_src_heatmaps.to(device), exmps_dst_heatmaps.to(device)

        xxx = _3D_downSmpl_multiprocessing(img_size, root, root_ver, src_name, dst_name)
        # xxx = [_3D_downSmpl(img_size, root, root_ver, src, dst) for src, dst in zip(src_name, dst_name)]

        texMask_src = [iitem[0] for iitem in xxx]
        texMask_dst = [iitem[1] for iitem in xxx]
            
        src_heatmaps = _numpy_to_tensor(np.array(texMask_src))
        src_heatmaps = src_heatmaps.permute(0, 3, 1, 2)
        dst_heatmaps = _numpy_to_tensor(np.array(texMask_dst))
        dst_heatmaps = dst_heatmaps.permute(0, 3, 1, 2)

        x_src, x_dst, n_src, n_dst = src_img.to(device), dst_img.to(device), src_heatmaps.to(device), dst_heatmaps.to(device)
        batch_size = x_src.size(0)

        #pdb.set_trace()
        n_src = nn.Upsample(size=(img_size, img_size), mode='bilinear')(n_src).type(torch.float32)
        n_dst = nn.Upsample(size=(img_size, img_size), mode='bilinear')(n_dst).type(torch.float32)
        x_fake, w = model_gen(x_src, n_src, n_dst)
        #x_fake = mask_fusion(x_fake, input_3dMask, x_dst, input_mms)

        eps = torch.rand(batch_size, 1).to(device)
        eps = eps.expand(-1, int(x_src.numel() / batch_size)).view_as(x_src)

        x_rand = eps * x_dst.detach() + (1 - eps) * x_fake.detach()
        x_rand.requires_grad_()
        x_rand = torch.cat([x_rand, n_dst], dim=1)
        loss_rand_x = model_dis(x_rand)

        grad_outputs = torch.ones(loss_rand_x.size())
        grads = autograd.grad(loss_rand_x, x_rand, grad_outputs=grad_outputs.to(device), create_graph=True)[0]
        loss_gp = torch.mean((grads.view(batch_size, -1).pow(2).sum(1).sqrt() - 1).pow(2))

        loss_real_x = model_dis(torch.cat([x_dst, n_dst], dim=1))
        loss_fake_x = model_dis(torch.cat([x_fake.detach(), n_dst], dim=1))
        loss_dis = loss_fake_x.mean() - loss_real_x.mean() + 10.0 * loss_gp

        # compute gradient and bp
        optim_dis.zero_grad()
        loss_dis.backward()
        optim_dis.step()

        dis_losses.update(float(loss_dis.item()))

        ######################
        # (2) Update G network
        ######################

        loss_fake_x = model_dis(torch.cat([x_fake, n_dst], dim=1))
        loss_gen = -loss_fake_x.mean() + 10000 * loss_l1(x_fake, x_dst) + 0.05 * loss_norm_l1(w)
   
        #pdb.set_trace()
        outputs_id_fea = net_r(nn.Upsample(size=(112, 112), mode='bilinear')(x_fake))
        #outputs_id = margin(outputs_id_fea, IDs.cuda().view(-1))
        fea_x_fake = outputs_id_fea
        fea_dst = net_r(nn.Upsample(size=(112, 112), mode='bilinear')(x_dst))
        id_loss = criterion_id(fea_x_fake, fea_dst)

        # compute gradient and bp
        optim_gen.zero_grad()
        (loss_gen+10000*id_loss).backward()
        optim_gen.step()

        gen_losses.update(float(loss_gen.item()))
        id_losses.update(float(id_loss.item()))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        output_trn = []
        output_val = []
        if (i % freq_disp == 0):
            exmps_n_src = nn.Upsample(size=(img_size, img_size), mode='bilinear')(exmps_n_src).type(torch.float32)
            exmps_n_dst = nn.Upsample(size=(img_size, img_size), mode='bilinear')(exmps_n_dst).type(torch.float32)
            exmps_x_fake, exmps_w = model_gen(exmps_x_src, exmps_n_src, exmps_n_dst)
            #exmps_x_fake = mask_fusion(exmps_x_fake, exmps_input_3dMask, exmps_x_dst, exmps_mms)
            nrow, exmps_x_out = cat_fivelet(exmps_x_src, exmps_n_src, exmps_x_dst, exmps_n_dst, exmps_x_fake)
            output_trn.append(torch_to_pil_image(exmps_x_out.detach()*0.5+0.5, nrow))
            output_trn[0].save('%s/train_'%output_dir + str(i+1) + '.jpg') #TODO to mod
            
            vals_n_src = nn.Upsample(size=(img_size, img_size), mode='bilinear')(vals_n_src).type(torch.float32)
            vals_n_dst = nn.Upsample(size=(img_size, img_size), mode='bilinear')(vals_n_dst).type(torch.float32)
            vals_x_fake, vals_w = model_gen(vals_x_src, vals_n_src, vals_n_dst)
            #vals_x_fake = mask_fusion(vals_x_fake, vals_input_3dMask, vals_x_dst, vals_mms)
            nrow, vals_x_out = cat_fivelet(vals_x_src, vals_n_src, vals_x_dst, vals_n_dst, vals_x_fake)
            output_val.append(torch_to_pil_image(vals_x_out.detach()*0.5+0.5, nrow))
            output_val[0].save('%s/val_'%output_dir + str(i+1) + '.jpg')#TODO to mod



        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss G: {loss_g:.4f} | Loss D: {loss_d: .4f} | id_loss: {id_loss: .4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss_g=gen_losses.avg,
            loss_d=dis_losses.avg,
            id_loss=id_losses.avg
        )
        bar.next()
        

    bar.finish()

    return gen_losses.avg, dis_losses.avg

def main():
    learning_rate = 0.00001
    optim_gen = optim.Adam(model_gen.parameters(), lr=learning_rate, betas=betas)
    optim_dis = optim.Adam(model_dis.parameters(), lr=1, betas=betas)
    normalize = NormalizeGjz(mean=127.5, std=128)  # may need optimization

    train_dataset = DDFADataset(
        root=root,
        img_size=img_size,
        filelists=filelists_train,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers,
                              shuffle=True, pin_memory=True, drop_last=True)

    val_dataset = DDFADataset(
        root=root,
        img_size=img_size,
        filelists=filelists_val,
        transform=transforms.Compose([ToTensorGjz(), normalize])
    )


    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers,
                              shuffle=True, pin_memory=True, drop_last=True)


    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            model_gen.load_state_dict(checkpoint['gen_state_dict'])
            model_dis.load_state_dict(checkpoint['dis_state_dict'])
            # import pdb; pdb.set_trace()
            # optim_gen.load_state_dict(checkpoint['gen_optim'])
            # optim_dis.load_state_dict(checkpoint['dis_optim'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
            out_dir_path = os.path.dirname(resume)
            logger = Logger(os.path.join(out_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(resume))
    else:
        start_epoch = 0
        out_dir_path = os.path.join(checkpoint_dir, datetime.datetime.now().isoformat()) # zyt mod

        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
            print('Make output dir: {}'.format(out_dir_path))

        logger = Logger(os.path.join(out_dir_path, 'log.txt'))
        logger.set_names(['Epoch', 'Train Loss G', 'Train Loss D'])

    for epoch in range(0, 0 + epochs):
        
        
        # train for one epoch
        if epoch in [10, 15, 18, 20]:
            # if epoch!=0: 
            learning_rate *= 0.1
            optim_gen = optim.Adam(model_gen.parameters(), lr=learning_rate, betas=betas)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 20, learning_rate))
        loss_gen, loss_dis = train(val_loader, train_loader, model_gen, model_dis, optim_gen, optim_dis, device)
        # append logger file
        logger.append([epoch + 20, loss_gen, loss_dis])
        
        if (epoch + 1) % snapshot == 0:
            # validate
            # validate(val_loader, model_gen, device, os.path.join(out_dir_path, 'epoch_{:04d}'.format(epoch + 1)))

            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 20,
                'gen_state_dict': model_gen.state_dict(),
                'dis_state_dict': model_dis.state_dict(),
                'gen_optim': optim_gen.state_dict(),
                'dis_optim': optim_dis.state_dict()
            }, checkpoint=out_dir_path)

    logger.close()
    logger.plot(['Train Loss G', 'Train Loss D'])
    savefig(os.path.join(out_dir_path, 'log.eps'))


if __name__ == '__main__':
    main()
