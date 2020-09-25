# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import argparse
import torch
import numpy as np
import pandas as pd
from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.engine.trainer import do_train
from ret_benchmark.losses import build_loss
from ret_benchmark.modeling import build_model
from ret_benchmark.solver import build_lr_scheduler, build_optimizer
from ret_benchmark.utils.logger import setup_logger
from ret_benchmark.utils.checkpoint import Checkpointer

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((8000, 512))
        labels = np.zeros(8000)
        #print(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
            #print(len(images))
            #print(k)
            embeddings[k:k+len(images)] = model.forward(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def train(cfg):
    logger = setup_logger(name='Train', level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    criterion = build_loss(cfg)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    train_loader = build_data(cfg, is_train=True)
    #for i, (input, target) in enumerate(train_loader):
     #   print("index:",i)
        #print("input:\n",len(input))
      #  print("target:\n",len(target))
    val_loader = build_data(cfg, is_train=False)

    logger.info(train_loader.dataset)
    logger.info(val_loader.dataset)

    arguments = dict()
    arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    checkpointer = Checkpointer(model, optimizer, scheduler, cfg.SAVE_DIR)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger
    )
    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    train_embeddings_cl_df = pd.DataFrame(train_embeddings_cl)
    train_embeddings_cl_df.to_csv("MSembedding_space.csv",index=False)
    


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a retrieval network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='config file',
        default=None,
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    train(cfg)
