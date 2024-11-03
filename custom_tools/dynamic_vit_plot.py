import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn as nn

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
import matplotlib.pyplot as plt
import cv2
warnings.filterwarnings("ignore")

unloader = transforms.ToPILImage()
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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
config_ic = dict(
            [('0', "none"),
             ('1', "none"),
             ('2', "none"),
             ('3', "DY_EE_HEAD"),
             ('4', "none"),
             ('5', "none"),
             ('6', "DY_EE_HEAD"),
             ('7', "none"),
             ('8', "none"),
             ('9', "DY_EE_HEAD"),
             ('10', "none"),
             ('11', "none"),
             ]
        )
num_classes = 200
model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size= None,
                              num_classes=num_classes,
                          config_ic=config_ic)

with open('/work/Anonymous/code/mmpretrain/work_dirs/config_diffrate_vit/last_checkpoint', 'r') as file:
    pretrained_dir = file.readline()
pretrained_dir = "/work/Anonymous/code/mmpretrain/work_dirs/config_diffrate_vit/best_accuracy_top1_epoch_147.pth"
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
model.ic_list[3].update_kept_token_number()
model.ic_list[6].update_kept_token_number()
model.ic_list[9].update_kept_token_number()
data_root = "/work/Anonymous/data/tiny-imagenet-200"
func = nn.Softmax(dim=1).cuda()
val_dataset = get_TinyImageNet(mode="val", data_root=data_root)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=8)
val_loader = tqdm.tqdm(val_loader, file=sys.stdout)

outputs_ = [[] for _ in range(3 + 1)]
outputs = []
with torch.no_grad():
    for step, data in enumerate(val_loader):
        images, labels = data
        # size = images.shape(0)
        images = images.cuda()
        labels = labels.cuda()
        pred_final, pred_ic = model(images)

        outputs_[0].append(func(pred_ic[0][1][0]))
        outputs_[1].append(func(pred_ic[1][1][0]))
        outputs_[2].append(func(pred_ic[2][1][0]))
        outputs_[3].append(func(pred_final))
outputs.append(torch.cat(outputs_[0], dim=0))
outputs.append(torch.cat(outputs_[1], dim=0))
outputs.append(torch.cat(outputs_[2], dim=0))
outputs.append(torch.cat(outputs_[3], dim=0))
val_pred = torch.stack(outputs)

output = []
probs_list = []
for p in range(1, 40):
    # print("*********************")
    _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
    probs = torch.exp(torch.log(_p) * torch.range(1, 3+1))
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
                                             num_workers=0)
model.mode = "inference"


def token_back_image(token):
    x = int(token / 14)
    y = int(token % 14)
    x = x * 16
    y = y * 16
    return y,x
def myplot(images, nums):
    unimage = unorm(images)
    unimage = unimage.mul(255).byte()
    unimage_ = unimage.cpu().numpy().transpose((1, 2, 0))
    unimage = cv2.cvtColor(unimage_, cv2.COLOR_RGB2BGR)
    for idx, image_idx in enumerate(image_idxs):
        x, y = token_back_image(image_idx)
        unimage = cv2.rectangle(unimage, (x, y), (x + 16, y + 16), (0, 0, 0), thickness=-1)
        if idx == nums:
            break
    unimage = cv2.cvtColor(unimage, cv2.COLOR_BGR2RGB)
    plt.imshow(unimage)
    plt.xticks([])
    plt.yticks([])
    return unimage_
with torch.no_grad():
    for (threshold_, prob_) in zip(output, probs_list):
        threshold = torch.zeros(12)
        threshold[3] = threshold_[0]
        threshold[6] = threshold_[1]
        threshold[9] = threshold_[2]
        model.threshold = threshold
        correct_num = 0
        flops = []
        for step, data in enumerate(test_loader):
            if step != 1031:
                continue
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            pred, exit_idx, keep_token, image_idxs = model(images)
            if exit_idx != -1:

                image_idxs = image_idxs[0].cpu().numpy()
                image_idxs = image_idxs[::-1]
                unimage = myplot(images[0].clone(), 70)
                # unimage = unorm(images[0])
                # unimage = unimage.mul(255).byte()
                # unimage = unimage.cpu().numpy().transpose((1, 2, 0))
                # unimage_ = cv2.cvtColor(unimage, cv2.COLOR_RGB2BGR)
                # for idx,image_idx in enumerate(image_idxs):
                #     x,y = token_back_image(image_idx)
                #     unimage = cv2.rectangle(unimage_, (x, y), (x + 16, y + 16), (255, 0, 0), 2)
                #     if idx == 70:
                #         break
                # unimage = cv2.cvtColor(unimage, cv2.COLOR_BGR2RGB)
                # unimage = cv2.resize(unimage, (64,64))
                # plt.axis('off')

                # plt.savefig("/work/Anonymous/data/tiny-imagenet-200/output/{}.png".format(step), dpi = 600)
                plt.savefig("/work/Anonymous/data/tiny-imagenet-200/output_dir/{}.png".format(step), dpi = 600, bbox_inches='tight')
                plt.imshow(unimage)
                plt.savefig("/work/Anonymous/data/tiny-imagenet-200/output_dir/{}_ori.png".format(step), dpi = 600,  bbox_inches='tight')
                if torch.argmax(pred) == labels:
                    correct_num += 1
        break
            # flops.append(model.flops())
        # print("threshold = ", threshold)
        # print("acc = ", correct_num / len(test_loader))
        # print("flops = ", np.mean(flops))
# print(self.dy_model.load_state_dict(weights_dict, strict=False))

