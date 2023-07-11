import argparse
import logging
from tqdm import tqdm

import torch
import yaml
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split
from torchvision import transforms

from model import get_pose_net 
from losses import CtdetLoss

from dataset import CTDetDataset

from utils import ctdet_decode, ctdet_post_process, merge_outputs

logger = logging.getLogger(__name__)

DATA_ROOT = "/mnt/data/coco"
SAVE_ROOT = "./checkpoints"


# class LitAutoEncoder(pl.LightningModule):
# 	def __init__(self):
# 		super().__init__()
#         self.model = PoseResNet()

# 	def forward(self, x):
# 		embedding = self.encoder(x)
# 		return embedding

# 	def configure_optimizers(self):
# 		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
# 		return optimizer

# 	def training_step(self, train_batch, batch_idx):
# 		x, y = train_batch
# 		x = x.view(x.size(0), -1)
# 		z = self.encoder(x)    
# 		x_hat = self.decoder(z)
# 		loss = F.mse_loss(x_hat, x)
# 		self.log('train_loss', loss)
# 		return loss

# 	def validation_step(self, val_batch, batch_idx):
# 		x, y = val_batch
# 		x = x.view(x.size(0), -1)
# 		z = self.encoder(x)
# 		x_hat = self.decoder(z)
# 		loss = F.mse_loss(x_hat, x)
# 		self.log('val_loss', loss)


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

    lr = 1e-3
    num_classes = 80
    max_obj = 128
    heads = {'hm': num_classes, 'reg': 2, 'wh': 2}
    head_conv = 64
    num_layers = 18 # 34 

    model = get_pose_net(num_layers, heads, head_conv)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss = CtdetLoss(cfg['loss'])
    model_with_loss = ModelWithLoss(model, loss)
    model_with_loss = model_with_loss.to(device=device)

    epochs = 1
    eval_interval = 2 

    for epoch in range(epochs):
        # train one epoch
        # model.train()
        # for batch in tqdm(train_loader):
        #     batch = {k:batch[k].to(device=device, non_blocking=True) for k in batch.keys() if k != 'meta'}
        #     output, loss, loss_stats = model_with_loss(batch)
        #     loss = loss.mean()
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     del batch, output, loss, loss_stats
        #     torch.cuda.empty_cache()

        # if the epoch is evaluation interval, evaluate the model
        if epoch % eval_interval == 0:
            model_with_loss.eval()
            with torch.no_grad():
              detections = []
              scale = 1 
              for batch in val_loader:

                  meta = batch['meta'].copy()
                  batch = {k:batch[k].to(device=device, non_blocking=True) for k in batch.keys() if k != 'meta'}
                  output, loss, loss_stats = model_with_loss(batch)
                  hm = output['hm'].sigmoid_()
                  wh = output['wh']
                  reg = output['reg']
                  dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=max_obj)
                  
                  torch.cuda.synchronize()
                  
                  dets = dets.detach().cpu().numpy()
                  batch_sz = dets.shape[0]
                  dets = dets.reshape(batch_sz, -1, dets.shape[2])
                  
                  mc = meta['mc'].numpy()
                  ms = meta['ms'].numpy() 
                  h = meta['out_height'].numpy()
                  w = meta['out_width'].numpy()
                  dets = ctdet_post_process(dets.copy(), mc, ms, h, w, num_classes=num_classes) 

                  for i in range(batch_sz):
                    for j in range(1, num_classes+1):
                      dets[i][j] = np.array(dets[i][j], dtype=np.float32).reshape(-1, 5)
                      dets[i][j][:, :4] /= scale

                    # results = merge_outputs(dets[i], num_classes=num_classes, max_per_image=max_obj)
                  
                    detections.append(dets)
                  break # NOMERGE
            
            import pdb; pdb.set_trace()
            # TODO: compare the outputs with the labels and compute mAp (COCO)
            del batch, output, loss, loss_stats
            torch.cuda.empty_cache()

        # if epoch % save_interval == 0:
        #     # save the model
        #     model.save(os.path.joint(SAVE_DIR, 'latest.pth'))


if __name__  == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--datadir", type=str, default="data")
    argparser.add_argument("--config", type=str, default="/home/art/code/point_tracker/config.yaml")
    
    args = argparser.parse_args()
    main(args)