import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from lib import dataset
from lib.dataset import train_collate_fn
from network import Segment
import logging.handlers
from lib.data_prefetcher import DataPrefetcher
import argparse
import time
from loss5 import SEG_LOSS
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.optim.lr_scheduler import SequentialLR

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def validate(net, val_loader):
    net.eval()
    mae, cnt, number = 0, 0, 256
    mean_pr, mean_re = 0, 0
    threshod = np.linspace(0, 1, number, endpoint=False)
    with torch.no_grad():
        for image, d, mask, (H, W), name, boundary in val_loader:
            image, d, mask = image.cuda().float(), d.cuda().float(), mask.cuda().float()
            out = net(image, d)
            pred = torch.sigmoid(out)
            cnt += 1
            mae += (pred - mask).abs().mean().item()
            precision = torch.zeros(number)
            recall = torch.zeros(number)
            for i in range(number):
                temp = (pred >= threshod[i]).float()
                precision[i] = (temp * mask).sum() / (temp.sum() + 1e-12)
                recall[i] = (temp * mask).sum() / (mask.sum() + 1e-12)
            mean_pr += precision
            mean_re += recall
    mae_val = mae / cnt
    mean_pr /= cnt
    mean_re /= cnt
    fscore = mean_pr * mean_re * (1 + 0.3) / (0.3 * mean_pr + mean_re + 1e-12)
    max_f_val = fscore.max().item()
    return {'mae': mae_val, 'max_f': max_f_val}
def train(image, depth, mask, boundary, optimizer,scheduler):
    net.train()
    while image is not None:
        out, out1_1, out2_1, out3_1, out4_1, out5_1, out2_2, out3_2, out4_2, out5_2 = net.forward(image, depth)
        seg_loss0 = SEG_LOSS(out, mask)
        seg_loss0_1 = SEG_LOSS(out1_1, mask)
        seg_loss1 = SEG_LOSS(out2_1, mask)
        seg_loss2 = SEG_LOSS(out3_1, mask)
        seg_loss3 = SEG_LOSS(out4_1, mask)
        seg_loss4 = SEG_LOSS(out5_1, mask)
        seg_loss5 = SEG_LOSS(out2_2, mask)
        seg_loss6 = SEG_LOSS(out3_2, mask)
        seg_loss7 = SEG_LOSS(out4_2, mask)
        seg_loss8 = SEG_LOSS(out5_2, mask)
        loss_E = seg_loss0_1
        loss_A = seg_loss1 + seg_loss5
        loss_B = seg_loss2 + seg_loss6
        loss_C = seg_loss3 + seg_loss7
        loss_D = seg_loss4 + seg_loss8
        loss = seg_loss0  + loss_E + loss_A + loss_B + loss_C + loss_D
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        image, depth, mask, boundary, gate_gt = prefetcher.next()
    return loss,

if __name__=='__main__':
    time1 = time.time()
    conf = argparse.ArgumentParser(description="train model")
    conf.add_argument("--back", type=str,default="pvt")
    conf.add_argument("--savepath", type=str)
    conf.add_argument("--lr", type=float, default=1e-4)
    conf.add_argument("--bz_size", type=int, default=24)
    conf.add_argument("--weight_decay", type=float, default=0)
    conf.add_argument("--epochs", type=int, default=50)
    conf.add_argument("--seed", type=int, default=1997)
    conf.add_argument("--gpu", type=str, default="9")

    args = conf.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(args.seed)
    DATA_PATH = "./USOD10k"
    args.warmup_epochs = 4
    args.start_factor =0.005

    args.savepath = os.path.join('./存储/', args.gpu, 'Weight')
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    log_dizhi = os.path.join('./存储/', args.gpu, 'log')
    if not os.path.exists(log_dizhi):
        os.makedirs(log_dizhi)
    logger = logging.getLogger('LOG_train')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_dizhi, "LOG_train.log")
    fh = logging.handlers.WatchedFileHandler(log_file_path, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("SEED:%s, gpu:%s" % (args.seed, args.gpu))
    logger.info("配置:{}".format(args))
    cfg = dataset.Config(datapath=DATA_PATH, savepath=args.savepath, mode='train', batch=args.bz_size, lr=args.lr,
                         epoch=args.epochs, train_scales=[256])
    data = dataset.RGBDData(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8, drop_last=True,
                        collate_fn=train_collate_fn)
    cfg_val = dataset.Config(datapath=DATA_PATH, mode='test')
    #cfg_val = dataset.Config(datapath=DATA_PATH, mode='val')
    data_val = dataset.RGBDData(cfg_val)
    loader_val = DataLoader(data_val, batch_size=1, shuffle=True, num_workers=0)
    net = Segment(backbone="pvtv2", aux_layers=True)
    net.cuda()
    logger.info(f"可训练的参数量: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.2f}M")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler_warmup = LinearLR(optimizer, start_factor=args.start_factor, total_iters=args.warmup_epochs)
    cosine_epochs = args.epochs - args.warmup_epochs
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine],
                             milestones=[args.warmup_epochs])
    best_mae = float('inf')
    best_max_f = 0.0
    for epoch in range(args.epochs):
        prefetcher = DataPrefetcher(loader, cnt=1)
        time9 = time.time()
        image, depth, mask, boundary, gate_gt = prefetcher.next()
        loss = train(image, depth, mask, boundary, optimizer, scheduler)
        time2_2 = time.time()
        time3_2 = time2_2 - time9
        total_minutes_2 = time3_2 // 60
        val_metrics = validate(net, loader_val)
        time2_3 = time.time()
        time3_3 = time2_3 - time2_2
        total_minutes_3 = time3_3 // 60
        val_mae = val_metrics['mae']
        val_max_f = val_metrics['max_f']
        logger.info(f'😘 Epoch: {epoch + 1:03d}/{args.epochs} | LR: {optimizer.param_groups[0]["lr"]:.6f} | Loss: {loss.item():.6f} | Val MAE: {val_mae:.6f} | Val Max-F: {val_max_f:.6f} | 一个epoch train的时间：{total_minutes_2}分钟 | 一个epoch val的时间：{total_minutes_3}分钟')
        if val_mae < best_mae:
            best_mae = val_mae
            best_max_f = val_max_f
            torch.save(net.state_dict(), cfg.savepath + '/best_model_' + str(epoch + 1))
            logger.info('💕💕 BEST!  : bestEpoch:{}  best_mae:{}  best_max_f:{}'.format(epoch + 1, best_mae, best_max_f))