import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn
import json
import tqdm
import sys
from pandas import DataFrame
import time
from TinyImageNet import get_TinyImageNet
import numpy as np
from collections import OrderedDict
sys.path.append("..")
from mmpretrain.models.classifiers.dynamic.vit_DiffRate import VisionTransformer
import warnings
warnings.filterwarnings("ignore")

def dynamic_eval_find_threshold(logits, p):

    n_stage, n_sample, c = logits.size()

    max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

    _, sorted_idx = max_preds.sort(dim=1, descending=True)

    filtered = torch.zeros(n_sample)
    T = torch.Tensor(n_stage).fill_(1e8)


    for k in range(n_stage - 1):
        acc, count = 0.0, 0
        out_n = math.floor(n_sample * p[k])
        for i in range(n_sample):
            ori_idx = sorted_idx[k][i]
            if filtered[ori_idx] == 0:
                count += 1
                if count == out_n:
                    T[k] = max_preds[k][ori_idx]
                    break
        filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))
    T[n_stage - 1] = -1e8  # accept all of the samples at the last stage
    return T
num_classes = 200
# ic3:   2.2ms
# ic6:   3.9ms
# ic9:   5.7ms
# final:   6.7ms

ic_time_consume = [2.2, 3.9,5.7,6.7]
ic_location = [3,6,9]
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
for ic in ic_location:
    config_ic[str(ic)] = "DY_EE_HEAD"

ic_time_consume_dict = dict()


# base
# ic_time_consume_dict[3] = 2.8
# ic_time_consume_dict[6] = 3.5
# ic_time_consume_dict[9] = 4.8
# ic_time_consume_dict[-1] = 7.5

# small
ic_time_consume_dict[3] = 2.2  # 2.6
ic_time_consume_dict[6] = 3.9  # 4.7
ic_time_consume_dict[9] = 5.7 # 6.8
ic_time_consume_dict[-1] = 6.7  # 7.5

# tiny
# ic_time_consume_dict[3] = 2.6  # 2.2
# ic_time_consume_dict[6] = 4.7
# ic_time_consume_dict[9] = 6.6
# ic_time_consume_dict[-1] = 7.6


result_dict = dict()
result_dict[-1] = dict([("right",[]),("fault",[]),("len", 0)])
for ic in ic_location:
    result_dict[ic] = dict([("right",[]),("fault",[]),("len", 0)])
# ic_num = 4
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
#                           patch_size=16,
#                           embed_dim=192,
#                           depth=12,
#                           num_heads=3,
#                           representation_size=None,
#                           num_classes=num_classes,
#                           config_ic=config_ic)



#  ******************************** deit small
# model = VisionTransformer(img_size=224,
#                                   patch_size=16,
#                                   embed_dim=384,
#                                   depth=12,
#                                   num_heads=6,
#                                   representation_size=None,
#                                   num_classes=num_classes,
#                                   config_ic=config_ic,
#                                   distilled=True)
# deit tiny
# model = VisionTransformer(img_size=224,
#                                   patch_size=16,
#                                   embed_dim=192,
#                                   depth=12,
#                                   num_heads=3,
#                                   representation_size=None,
#                                   num_classes=num_classes,
#                                   config_ic=config_ic,
#                                   distilled=True)
# with open('/work/Anonymous/code/mmpretrain/work_dirs/config_diffrate_vit/last_checkpoint', 'r') as file:
#     pretrained_dir = file.readline()

pretrained_dir = "/media/data/Anonymous/code/mmpretrain/work_dirs/config_diffrate_vit/epoch_99.pth"
# pretrained_dir = "/work/Anonymous/code/mmpretrain/work_dirs/config_diffrate_vit/weights/epoch_300_deit_tiny.pth"
# pretrained_dir = "/work/Anonymous/code/mmpretrain/work_dirs/config_diffrate_vit/weights/epoch_300_vit_base.pth"
weights_dict_ = torch.load(pretrained_dir, map_location="cpu")
weights_dict = weights_dict_["state_dict"]
checkpoint = OrderedDict()

for item in weights_dict:
    if "dy_model" in item:
        checkpoint[item.replace('dy_model.', '')] = weights_dict[item]

print(model.load_state_dict(checkpoint, strict=True))

model = model.cuda()
model.eval()
model.mode = "train"
for idx, ic in enumerate(model.ic_list):
    if hasattr(ic, 'kept_token_number'):
        ic.update_kept_token_number()

# model.ic_list[3].update_kept_token_number()
# model.ic_list[6].update_kept_token_number()
# model.ic_list[9].update_kept_token_number()
# data_root = "/media/data/Anonymous/data/tiny-imagenet-200"
data_root = "/work/Anonymous/data/tiny-imagenet-200"

func = nn.Softmax(dim=1).cuda()
val_dataset = get_TinyImageNet(mode="val", data_root=data_root)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8)
val_loader = tqdm.tqdm(val_loader, file=sys.stdout)

outputs_ = [[] for _ in range(len(ic_location) + 1)]
outputs = []
with torch.no_grad():
    for step, data in enumerate(val_loader):
        images, labels = data
        # size = images.shape(0)
        images = images.cuda()
        labels = labels.cuda()
        pred_final, pred_ics = model(images)
        for idx, pred_ic in enumerate(pred_ics):
            outputs_[idx].append(func(pred_ic[1][0]))
        # outputs_[0].append(func(pred_ic[0][1][0]))
        # outputs_[1].append(func(pred_ic[1][1][0]))
        # outputs_[2].append(func(pred_ic[2][1][0]))
        outputs_[-1].append(func(pred_final))
# outputs.append(torch.cat(outputs_[0], dim=0))
# outputs.append(torch.cat(outputs_[1], dim=0))
# outputs.append(torch.cat(outputs_[2], dim=0))
# outputs.append(torch.cat(outputs_[3], dim=0))
for item in outputs_:
    outputs.append(torch.cat(item, dim=0))
val_pred = torch.stack(outputs)

output = []
probs_list = []
for p in range(1, 40):
    # print("*********************")
    _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
    probs = torch.exp(torch.log(_p) * torch.range(1, len(ic_location)+1))
    probs /= probs.sum()
    T = dynamic_eval_find_threshold(
        val_pred,  probs)
    output.append(T)
    probs_list.append(probs)

test_dataset = get_TinyImageNet(mode="val", data_root=data_root)



test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=1)
model.mode = "inference"

write_dict = dict()
write_dict["time"] = []
write_dict["flops"] = []
write_dict["acc"] = []
write_dict["acc_ic3"] = []
write_dict["acc_ic6"] = []
write_dict["acc_ic9"] = []
with torch.no_grad():
    # 预热model
    model.threshold = torch.zeros(12)
    for _ in range(50):
        input = torch.randn((1, 3, 224, 224)).cuda()
        model(input)
    for index, (threshold_, prob_) in enumerate(zip(output, probs_list)):
        if index > 30:
            break
        threshold = torch.zeros(12)
        for idx, ic in enumerate(ic_location):
            threshold[ic] = threshold_[idx]
        # threshold[3] = threshold_[0]
        # threshold[6] = threshold_[1]
        # threshold[9] = threshold_[2]
        model.threshold = threshold
        correct_num = 0
        flops = []
        # duration = []
        for step, data in enumerate(test_loader):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            start_time = time.time()
            pred, exit_idx, keep_token, image_idxs = model(images)
            torch.cuda.synchronize()
            end_time = time.time()
            if torch.argmax(pred) == labels:
                correct_num += 1
                result_dict[exit_idx]["right"].append(step)
            else:
                result_dict[exit_idx]["fault"].append(step)
            result_dict[exit_idx]["len"] += 1
            duration_ = (end_time - start_time)
            # if exit_idx == 3:
            #     duration.append(duration_)
                # print("duration_ = ", duration_)

            flops.append(model.flops())
        # print("mean time = ", np.mean(duration))
        total_time = 0.0
        for ic in ic_location:
            total_time += result_dict[ic]["len"] * ic_time_consume_dict[ic]
        total_time += result_dict[-1]["len"] * ic_time_consume_dict[-1]
        total_time = total_time / len(test_loader)
        acc = correct_num / len(test_loader)
        flops = np.mean(flops)
        print("threshold = ", threshold)
        print("acc = ", acc)
        print("flops = ", flops)
        print("total_time = ", total_time)
        write_dict["time"].append(total_time)
        write_dict["flops"].append(flops)
        write_dict["acc"].append(acc)
        write_dict["acc_ic3"].append((len(result_dict[3]["right"]) * 1.0) / result_dict[3]["len"])
        write_dict["acc_ic6"].append((len(result_dict[6]["right"]) * 1.0) / result_dict[6]["len"])
        write_dict["acc_ic9"].append((len(result_dict[9]["right"]) * 1.0) / result_dict[9]["len"])
        for key in result_dict:
            result_dict[key] = dict([("right",[]),("fault",[]),("len", 0)])
# 将数据写入 JSON 文件
with open('data.json', 'w') as file:
    json.dump(write_dict, file)

# print(self.dy_model.load_state_dict(weights_dict, strict=False))