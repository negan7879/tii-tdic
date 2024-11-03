import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn
from functools import partial
import tqdm
import sys
from pandas import DataFrame
import time
from TinyImageNet import get_TinyImageNet
import numpy as np
from collections import OrderedDict
sys.path.append("..")
from mmpretrain.models.classifiers.dynamic.vit_DiffRate import VisionTransformer
from mmpretrain.models.classifiers.dynamic.evit import EViT
from mmpretrain.models.classifiers.dynamic.pvt import PyramidVisionTransformer
from mmpretrain.models.classifiers.dynamic.t2t_vit import T2T_ViT
from mmpretrain.models.classifiers.dynamic.levit import LeViT_128,LeViT_192,LeViT_256


import warnings
from torchprofile import profile_macs

from mmpretrain.evaluation import Accuracy
warnings.filterwarnings("ignore")

num_classes = 200
config_ic = dict(
            [('0', "none"),
             ('1', "none"),
             ('2', "none"),
             ('3', "none"),
             ('4', "none"),
             ('5', "none"),
             ('6', "none"),
             ('7', "none"),
             ('8', "none"),
             ('9', "none"),
             ('10', "none"),
             ('11', "none"),
             ]
        )
# base
# model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size= None,
#                               num_classes=num_classes,
#                           config_ic=config_ic)
# small
model = VisionTransformer(img_size=224,
                          patch_size=16,
                          embed_dim=384,
                          depth=12,
                          num_heads=6,
                          representation_size=None,
                          num_classes=num_classes,
                          config_ic=config_ic)
# tiny
# model = VisionTransformer(img_size=224,
#                                           patch_size=16,
#                                           embed_dim=192,
#                                           depth=12,
#                                           num_heads=3,
#                                           representation_size=None,
#                                           num_classes=num_classes,
#                                           config_ic=config_ic)


# deit base
#
# model = VisionTransformer(img_size=224,
#                             patch_size=16,
#                             embed_dim=768,
#                             depth=12,
#                             num_heads=12,
#                             representation_size=None,
#                             num_classes=num_classes,
#                             config_ic=config_ic,
#                             distilled=True)
# deit small
# model = VisionTransformer(img_size=224,
#                           patch_size=16,
#                           embed_dim=384,
#                           depth=12,
#                           num_heads=6,
#                           representation_size=None,
#                           num_classes=num_classes,
#                           config_ic=config_ic,
#                           distilled=True)
# deit tiny
# model = VisionTransformer(img_size=224,
#                           patch_size=16,
#                           embed_dim=192,
#                           depth=12,
#                           num_heads=3,
#                           representation_size=None,
#                           num_classes=num_classes,
#                           config_ic=config_ic,
#                           distilled=True)


# small vit evit
base_rate = 0.5
keep_rate = (1, 1, 1, base_rate, 1, 1, base_rate, 1, 1, base_rate, 1, 1)

# base
# model =  EViT(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=768, depth=12,
#                      num_heads=12, mlp_ratio=4,
#                      qkv_bias=True, representation_size=None, distilled=False, drop_rate=0.0, attn_drop_rate=0.0,
#                      drop_path_rate=0.1,
#                      fuse_token=True, keep_rate=keep_rate)
# # model = EViT(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=384, depth=12,
# #                              num_heads=6, mlp_ratio=4,
# #                              qkv_bias=True, representation_size=None, distilled=False, drop_rate=0.0,
# #                              attn_drop_rate=0.0,
# #                              drop_path_rate=0.1,
# #                              fuse_token=True, keep_rate=keep_rate)
# tiny
# model = EViT(img_size=224, patch_size=16, in_chans=3, num_classes=num_classes, embed_dim=192, depth=12,
#                              num_heads=3, mlp_ratio=4,
#                              qkv_bias=True, representation_size=None, distilled=False, drop_rate=0.0,
#                              attn_drop_rate=0.0,
#                              drop_path_rate=0.1,
#                              fuse_token=True, keep_rate=keep_rate)



# pvt small
# model = PyramidVisionTransformer(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_classes = num_classes)
# pvt medium
# model = PyramidVisionTransformer(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             num_classes=num_classes)

# pvt tiny
# model = PyramidVisionTransformer(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True,
#             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], num_classes=num_classes)


# t2t 14
# model = T2T_ViT(num_classes=num_classes, tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3)
# model = T2T_ViT(num_classes=num_classes, tokens_type='transformer',embed_dim=448, depth=19, num_heads=7, mlp_ratio=3.)
# model = T2T_ViT(num_classes=num_classes, tokens_type='transformer',embed_dim=512, depth=24, num_heads=8, mlp_ratio=3.,)


# levit
# model = LeViT_128(num_classes=num_classes, distillation=False)
# model = LeViT_192(num_classes=num_classes, distillation=True)
# model = LeViT_256(num_classes=num_classes, distillation=True)
# ****************************************************

model.eval()
x = torch.rand(1, 3, 224, 224)
macs = profile_macs(model, x) * 1e-9
print("macs = ", macs)

# with open('/media/data/Anonymous/code/mmpretrain/work_dirs/config_finetune/last_checkpoint', 'r') as file:
#     pretrained_dir = file.readline()


# base vit weight
# pretrained_dir = "/work/Anonymous/data/weights/vit_base_finetune_tinyimagenet.pth"
# small vit weight
pretrained_dir = "/work/Anonymous/data/weights/vit_small_finetune_tinyimagenet.pth"
# tiny vit weight
# pretrained_dir = "/work/Anonymous/data/weights/vit-tiny_finetune_tinyimagenet.pth"


# deit
# pretrained_dir = "/work/Anonymous/data/weights/deit_tiny_tinyimagenet.pth"
# pretrained_dir = "/work/Anonymous/data/weights/deit_base_tinyimagenet.pth"

# levit
# pretrained_dir = "/media/data/Anonymous/code/mmpretrain/work_dirs/config_finetune/epoch_100.pth"


# pvt
# pretrained_dir = "/work/Anonymous/data/weights/pvt_medium_tinyimagenet.pth"

# small vit evit weight
# pretrained_dir = "/work/Anonymous/data/weights/vit_evit_small-0.7.pth"
# pretrained_dir = "/work/Anonymous/data/weights/vit_evit_small_0.5.pth"
# pretrained_dir = "/work/Anonymous/data/weights/vit_evit_tiny_0.7.pth"

# t2t
# pretrained_dir = "/work/Anonymous/code/mmpretrain/work_dirs/config_finetune/weights/t2t_14_tinyimaget.pth"
# pretrained_dir = "/work/Anonymous/code/mmpretrain/work_dirs/config_finetune/weights/t2t_19_tinyimaget.pth"
# pretrained_dir = "/work/Anonymous/code/mmpretrain/work_dirs/config_finetune/weights/t2t_24_tinyimagenet.pth"

weights_dict_ = torch.load(pretrained_dir, map_location="cpu")
weights_dict = weights_dict_["state_dict"]
checkpoint = OrderedDict()
for item in weights_dict:
    if "dy_model" in item:
        checkpoint[item.replace('dy_model.', '')] = weights_dict[item]

print(model.load_state_dict(checkpoint, strict=True))

model = model.cuda()
model.eval()
model.mode = "val"

data_root = "/media/data/Anonymous/data/tiny-imagenet-200"
# data_root = "/work/Anonymous/data/tiny-imagenet-200"
func = nn.Softmax(dim=1).cuda()
val_dataset = get_TinyImageNet(mode="val", data_root=data_root)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1)
val_loader = tqdm.tqdm(val_loader, file=sys.stdout)

# outputs_ = [[] for _ in range(3 + 1)]
# outputs = []
acc1_list = []
acc5_list = []
# flops = []
for _ in range(50):
    input = torch.randn((1, 3, 224, 224)).cuda()
    model(input)
with torch.no_grad():
    times = []
    for step, data in enumerate(val_loader):
        images, labels = data
        # size = images.shape(0)
        images = images.cuda()
        labels = labels.cuda()
        start = time.time()
        pred_final= model(images)
        end = time.time()
        elapse = end - start
        if isinstance(pred_final, tuple):
            pred_final = pred_final[0]
        torch.cuda.synchronize()

        times.append(elapse * 1000)
        acc_1, acc_5 = Accuracy.calculate(pred_final, labels, topk=(1, 5))
        acc1_list.append(acc_1[0].cpu().numpy())
        acc5_list.append(acc_5[0].cpu().numpy())
        # flops.append(model.flops())
    print("acc top-1 = ", np.mean(acc1_list))
    print("acc top-5 = ", np.mean(acc5_list))
    print("single time  = ", np.mean(times))
    # print("flops = ", np.mean(flops) / (1e9))
    pass
#         outputs_[0].append(func(pred_ic[0][1][0]))
#         outputs_[1].append(func(pred_ic[1][1][0]))
#         outputs_[2].append(func(pred_ic[2][1][0]))
#         outputs_[3].append(func(pred_final))
# outputs.append(torch.cat(outputs_[0], dim=0))
# outputs.append(torch.cat(outputs_[1], dim=0))
# outputs.append(torch.cat(outputs_[2], dim=0))
# outputs.append(torch.cat(outputs_[3], dim=0))
# val_pred = torch.stack(outputs)

# output = []
# probs_list = []
# for p in range(1, 40):
#     # print("*********************")
#     _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
#     probs = torch.exp(torch.log(_p) * torch.range(1, 3+1))
#     probs /= probs.sum()
#     T = dynamic_eval_find_threshold(
#         val_pred,  probs)
#     output.append(T)
#     probs_list.append(probs)

# test_dataset = get_TinyImageNet(mode="val", data_root=data_root)

#
#
# test_loader = torch.utils.data.DataLoader(test_dataset,
#                                              batch_size=1,
#                                              shuffle=False,
#                                              pin_memory=False,
#                                              num_workers=1)
# model.mode = "inference"
#
#
# with torch.no_grad():
#     for (threshold_, prob_) in zip(output, probs_list):
#         threshold = torch.zeros(12)
#         threshold[3] = threshold_[0]
#         threshold[6] = threshold_[1]
#         threshold[9] = threshold_[2]
#         model.threshold = threshold
#         correct_num = 0
#         flops = []
#         for step, data in enumerate(test_loader):
#             images, labels = data
#             images = images.cuda()
#             labels = labels.cuda()
#             pred, exit_idx, keep_token, image_idxs = model(images)
#             if torch.argmax(pred) == labels:
#                 correct_num += 1
#             flops.append(model.flops())
#         print("threshold = ", threshold)
#         print("acc = ", correct_num / len(test_loader))
#         print("flops = ", np.mean(flops))
# print(self.dy_model.load_state_dict(weights_dict, strict=False))