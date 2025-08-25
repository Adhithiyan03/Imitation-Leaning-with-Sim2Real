# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython


def get_args():
    """
    Redone to not use argparse
    """
    args = argparse.Namespace()

    # parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # parser.add_argument('--lr', default=1e-4, type=float)  # will be overridden
    # parser.add_argument('--lr_backbone', default=1e-5, type=float)  # will be overridden
    # parser.add_argument('--batch_size', default=2, type=int)  # not used
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--epochs', default=300, type=int)  # not used
    # parser.add_argument('--lr_drop', default=200, type=int)  # not used
    # parser.add_argument('--clip_max_norm', default=0.1, type=float,  # not used
    #                     help='gradient clipping max norm')

    # Set transformer detector
    args.lr = 1e-4
    args.lr_backbone = 1e-5
    args.batch_size = 2
    args.weight_decay = 1e-4
    args.epochs = 300
    args.lr_drop = 200
    args.clip_max_norm = 0.1

    # Model parameters
    # * Backbone
    # parser.add_argument('--backbone', default='resnet18', type=str,  # will be overridden
    #                     help="Name of the convolutional backbone to use")
    # parser.add_argument('--dilation', action='store_true',
    #                     help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
    #                     help="Type of positional embedding to use on top of the image features")
    # parser.add_argument('--camera_names', default=[], type=list,  # will be overridden
    #                     help="A list of camera names")
    args.backbone = 'resnet18'
    args.dilation = False
    args.position_embedding = 'sine'
    args.camera_names = []

    # # * Transformer
    # parser.add_argument('--enc_layers', default=4, type=int,  # will be overridden
    #                     help="Number of encoding layers in the transformer")
    # parser.add_argument('--dec_layers', default=6, type=int,  # will be overridden
    #                     help="Number of decoding layers in the transformer")
    # parser.add_argument('--dim_feedforward', default=2048, type=int,  # will be overridden
    #                     help="Intermediate size of the feedforward layers in the transformer blocks")
    # parser.add_argument('--hidden_dim', default=256, type=int,  # will be overridden
    #                     help="Size of the embeddings (dimension of the transformer)")
    # parser.add_argument('--dropout', default=0.1, type=float,
    #                     help="Dropout applied in the transformer")
    # parser.add_argument('--nheads', default=8, type=int,  # will be overridden
    #                     help="Number of attention heads inside the transformer's attentions")
    # parser.add_argument('--num_queries', default=400, type=int,  # will be overridden
    #                     help="Number of query slots")
    # parser.add_argument('--pre_norm', action='store_true')
    args.enc_layers = 4
    args.dec_layers = 6
    args.dim_feedforward = 2048
    args.hidden_dim = 256
    args.dropout = 0.1
    args.nheads = 8
    args.num_queries = 400
    args.pre_norm = False

    # # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                     help="Train segmentation head if the flag is provided")
    args.masks = False

    return args


def build_ACT_model_and_optimizer(args_override):
    args = get_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer
