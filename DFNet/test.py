import os
import sys
sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
from network import Segment
import time
import logging as logger
import argparse
from tqdm import tqdm
import logging.handlers

DATASETS = ['./USOD10k', './USOD', ]

class Test(object):
    def __init__(self, conf, Dataset, datapath):
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s" % self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.RGBDData(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=0)
        self.net = Segment(backbone="pvtv2",  aux_layers=True)
        self.net.train(False)
        print(f"可训练的参数量: {sum(p.numel() for p in self.net.parameters() if p.requires_grad) / 1e6:.2f}M")
        self.net.load_state_dict(torch.load(conf.model))
        self.net.cuda()
        self.net.eval()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, d, mask, (H, W), name, boundary in self.loader:
                image, d, mask = image.cuda().float(), d.cuda().float(), mask.cuda().float()
                start_time = time.time()
                out = self.net(image, d)
                pred = torch.sigmoid(out)
                torch.cuda.synchronize()
                end_time = time.time()
                cost_time += end_time - start_time
                cnt += 1
                mae += (pred - mask).abs().mean()
                precision = torch.zeros(number)
                recall = torch.zeros(number)
                for i in range(number):
                    temp = (pred >= threshod[i]).float()
                    precision[i] = (temp * mask).sum() / (temp.sum() + 1e-12)
                    recall[i] = (temp * mask).sum() / (mask.sum() + 1e-12)
                mean_pr += precision
                mean_re += recall
                fscore = mean_pr * mean_re * (1 + 0.3) / (0.3 * mean_pr + mean_re + 1e-12)
                if cnt % 20 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    print('MAE=%.6f, F-score=%.6f, fps=%.4f' % (mae / cnt, fscore.max() / cnt, fps))
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f' % (self.datapath, mae / cnt, fscore.max() / cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)

    def save(self):
        with torch.no_grad():
            for image, d, mask, (H, W), name, boundary in tqdm(self.loader):
                image, d = image.cuda().float(), d.cuda().float()
                out = self.net(image, d)
                out = F.interpolate(out, size=(H, W), mode='bilinear')
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                head = './推理结果/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], np.uint8(pred))

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description="train model")
    conf.add_argument("--back", type=str, default="pvtv2")
    conf.add_argument("--gpu", type=str, default="0")
    args = conf.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    log_dizhi = os.path.join('./推理结果/', 'log')
    if not os.path.exists(log_dizhi):
        os.makedirs(log_dizhi)
    args.model = './best_model'
    logger = logging.getLogger('LOG_test')
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_dizhi, "LOG_test.log")
    fh = logging.handlers.WatchedFileHandler(log_file_path, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("配置:{}".format(args))
    for e in DATASETS:
        t = Test(args, dataset, e)
        t.accuracy()
        t.save()