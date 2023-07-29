import cv2 
import argparse
import logging
from tqdm import tqdm

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path

from model import get_pose_net 
from losses import CtdetLoss

from dataset import CTDetDataset
from torch.utils.tensorboard import SummaryWriter


from utils import ctdet_decode, COCOEvaluator

logger = logging.getLogger(__name__)

#DATA_ROOT = "/mnt/data/coco"
DATA_ROOT = "/home/art/data_tmp/coco"
SAVE_ROOT = Path("./checkpoints")

# default `log_dir` is "runs" - we'll be more specific here
tsb_writer = SummaryWriter('./tensorboard')


class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats             


def main(args):
    """
    1. Make the dataloaders
    2. Make the model
    3. Train the model
    4. Evaluate the model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # read the config
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # make the dataloaders
    train_dataset = CTDetDataset(cfg['dataset'], DATA_ROOT, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=True,
                                               num_workers=cfg['num_workers'],
                                               pin_memory=True
    )
    
    val_dataset = CTDetDataset(cfg['dataset'], DATA_ROOT, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=cfg['batch_size'],
                                               shuffle=False,
                                               num_workers=cfg['num_workers'],
                                               pin_memory=True
    )

    # invariants
    # TODO: make these configurable
    num_classes = 80
    heads = {'hm': num_classes, 'reg': 2, 'wh': 2}
    head_conv = 64
    
    max_obj = cfg['max_obj']
    num_layers = cfg['num_layers'] 

    model = get_pose_net(num_layers, heads, head_conv)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    loss_f = CtdetLoss(cfg['loss'])
    model_with_loss = ModelWithLoss(model, loss_f)
    model_with_loss = model_with_loss.to(device=device)

    epochs = 60
    eval_interval = 1
    save_interval = 4

    for epoch in range(epochs):
        # train one epoch
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader):
            batch = {k:batch[k].to(device=device, non_blocking=True) for k in batch.keys() if k != 'meta'}
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            del batch, output, loss, loss_stats
            torch.cuda.empty_cache()

        # log the running loss
        tsb_writer.add_scalar('training loss', running_loss / len(train_dataset), epoch)

        # if the epoch is evaluation interval, evaluate the model
        if epoch % eval_interval == 0:
            model_with_loss.eval()
            evaluator = COCOEvaluator(val_dataset.annot_path, val_dataset.cat_ids)
            with torch.no_grad():
              for batch in tqdm(val_loader):
                  meta = batch['meta']
                  batch = {k:batch[k].to(device=device, non_blocking=True) for k in batch.keys() if k != 'meta'}
                  output, loss, loss_stats = model_with_loss(batch)
                  hm = output['hm'].sigmoid_()
                  wh = output['wh']
                  reg = output['reg']
                  dets = ctdet_decode(hm, wh, reg, K=max_obj)
                  torch.cuda.synchronize()
                  
                  dets = dets.detach().cpu().numpy()
                  batch_sz = dets.shape[0]
                 
                  # scale up the detections
                  for i in range(batch_sz):
                    out_scale = meta['out_scale'][i].numpy()
                    img_dets = dets[i]
                    img_dets = img_dets[img_dets[:, 4] > 0.]
                    img_dets[:, [0, 2]] = img_dets[:, [0, 2]] / out_scale[0]
                    img_dets[:, [1, 3]] = img_dets[:, [1, 3]] / out_scale[1]
                    evaluator.add_img_dets(meta['img_id'][i], img_dets)
            
            mAp, mAp50 = evaluator.evaluate()
            
            # add to the tensorboard 
            tsb_writer.add_scalar('validation mAP', mAp, epoch)
            tsb_writer.add_scalar('validation mAP@50', mAp50, epoch)

            del batch, output, loss, loss_stats
            torch.cuda.empty_cache()
        
        # save the model
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        model_path = SAVE_ROOT / 'latest.pth'
        torch.save(model_with_loss.model.state_dict(), model_path)
        if epoch % save_interval == 0:
            model_path = SAVE_ROOT / f'epoch_{epoch}.pth'
            torch.save(model_with_loss.model.state_dict(), model_path)


if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--datadir", type=str, default="data")
    argparser.add_argument("--config", type=str, default="/home/art/code/point_tracker/config.yaml")
    
    args = argparser.parse_args()
    main(args)