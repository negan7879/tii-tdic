import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import timm.optim.optim_factory as optim_factory
import argparse
import datetime
import json
import numpy as np
import sys
import os
import time
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cosine
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from functools import partial
import torchvision.datasets as datasets
from scipy.stats import gaussian_kde
from TinyImageNet import get_TinyImageNet
import numpy as np
from collections import OrderedDict
sys.path.append("..")
from mmpretrain.models.classifiers.dynamic.vit_DiffRate import VisionTransformer
from mmpretrain.models.classifiers.dynamic.t2t_vit import T2T_ViT
from mmpretrain.models.classifiers.dynamic.pvt import PyramidVisionTransformer
def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(3407)
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
def similarity_pairs(tokens):
    attention_scores = torch.matmul(tokens, tokens.transpose(-1, -2)) / torch.sqrt(torch.tensor(tokens.size(-1), dtype=torch.float))

    # 将对角线上的元素设置为 0，因为一个 token 和它自己的相似度应该为 1，这里将它们设为 0，不参与后续计算
    attention_scores = attention_scores * (1 - torch.eye(tokens.size(1), device=attention_scores.device))

    # 计算 token 之间的相似度标量
    avg_similarity = F.softmax(attention_scores, dim=-1).sum(-1).mean()

    return avg_similarity.item()
def distributions_similarity(batch_tokens):
    bs = len(batch_tokens)
    sim = []
    # 将概率分布转换为概率密度函数

    for idx in range(bs):
        tokens = batch_tokens[idx]
        pdfs = []
        for token in tokens:
            vector_ = token.data.cpu().numpy()
            # 将 token 转换为向量（例如使用 VIT 模型得到的嵌入向量）
            kde = gaussian_kde(vector_)
            x = np.linspace(min(vector_), max(vector_), num=100)
            # 计算估计的概率密度函数值
            pdf = kde.evaluate(x)
            pdfs.append(pdf)
    # for dist in distributions:
    #     pdf = # 将分布转换为概率密度函数
    #     pdfs.append(pdf)
    # 计算 Jensen-Shannon 距离并返回平均值
        sim_matrix = np.zeros((len(pdfs), len(pdfs)))
        for i in range(len(pdfs)):
            for j in range(i+1, len(pdfs)):
                sim_matrix[i][j] = jensenshannon(pdfs[i], pdfs[j])
    # 返回平均相似度
    #     sim.append(np.mean(sim_matrix[:,11]))

        sim.append(np.mean(sim_matrix))
    return sim
def tokens_similarity(batch_tokens):
    # 将 token 转换为向量，以便进行余弦相似度计算
    # batch_tokens = batch_tokens.permute(0,2,1)
    #
    # def cos_sim(x):
    #     x = F.normalize(x, p=2, dim=1, eps=1e-8)
    #     cos_sim = torch.matmul(x.transpose(1, 2), x)
    #     return torch.abs(cos_sim)
    #
    # inputs_cos = cos_sim(batch_tokens)
    # sim = torch.mean(inputs_cos, dim=(1,2))
    # return sim
    bs = len(batch_tokens)
    sim = []
    for idx in range(bs):
        tokens = batch_tokens[idx]
        vectors = []
        for token in tokens[0:]:
            vector = token.data.cpu().numpy() # 将 token 转换为向量（例如使用 VIT 模型得到的嵌入向量）
            vectors.append(vector)
        # 计算余弦相似度并返回平均值
        sim_matrix = np.zeros((len(vectors), len(vectors)))
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                # dot_product = np.dot(vectors[i], vectors[j])
                # norm_i = np.linalg.norm(vectors[i])
                # norm_j = np.linalg.norm(vectors[j])
                # sim_matrix[i][j] = dot_product / (norm_i * norm_j + 1e-08)
                sim_matrix[i][j] = cosine(vectors[i],vectors[j])
                # pass
                # sim_matrix[i][j] =  F.cosine_similarity(torch.from_numpy(vectors[i]).unsqueeze(0), torch.from_numpy(vectors[j]).unsqueeze(0))
        sim.append(np.mean(sim_matrix))
        # sim.append(np.mean(sim_matrix[:,11]))
    # 返回平均相似度
    return sim
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

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
#                               num_classes=1000,
#                           config_ic=config_ic)

# small
# model = VisionTransformer(img_size=224,
#                           patch_size=16,
#                           embed_dim=384,
#                           depth=12,
#                           num_heads=6,
#                           representation_size=None,
#                           num_classes=1000,
#                           config_ic=config_ic)

# model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=192,
#                               depth=12,
#                               num_heads=3,
#                               representation_size=None,
#                               num_classes=1000,
#                               config_ic=config_ic)
# model = T2T_ViT(num_classes=1000, tokens_type='transformer',embed_dim=448, depth=19, num_heads=7, mlp_ratio=3.)
# model = T2T_ViT(num_classes=1000, tokens_type='transformer',embed_dim=512, depth=24, num_heads=8, mlp_ratio=3.,)

model = PyramidVisionTransformer(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            num_classes=1000)
# pretrained_dir = "/work/Anonymous/data/weights/vit_base_finetune_tinyimagenet.pth"
# pretrained_dir = "/work/Anonymous/data/weights/deit_base_patch16_224-b5f2ef4d.pth"
# pretrained_dir = "/work/Anonymous/data/weights/deit_small_patch16_224-cd65a155.pth"
# pretrained_dir = "/work/Anonymous/data/weights/deit_tiny_patch16_224-a1311bcf.pth"
# pretrained_dir = "/work/Anonymous/data/weights/82.6_T2T_ViTt_24.pth.tar"
pretrained_dir = "/work/Anonymous/data/weights/pvt_medium.pth"


weights_dict_ = torch.load(pretrained_dir, map_location="cpu")
# weights_dict = weights_dict_["model"]
# weights_dict = weights_dict_["state_dict_ema"]
weights_dict = weights_dict_
# weights_dict = weights_dict_["state_dict"]
# checkpoint = OrderedDict()
# for item in weights_dict:
#     if "dy_model" in item:
#         checkpoint[item.replace('dy_model.', '')] = weights_dict[item]

print(model.load_state_dict(weights_dict, strict=True))

model = model.cuda()

# data_root = "/work/Anonymous/data/tiny-imagenet-200"
data_root = "/work/data/imagenet/"
val_dataset = get_TinyImageNet(mode="val", data_root=data_root)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8)

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.feature = output

    def remove(self):
        self.hook.remove()

hooks = []

# for m in model.blocks:
#     hooks.append(AttentionMap(m.attn.matmul))
for m in model.block1:
    hooks.append(AttentionMap(m.attn.matmul))
for m in model.block2:
    hooks.append(AttentionMap(m.attn.matmul))
for m in model.block3:
    hooks.append(AttentionMap(m.attn.matmul))
for m in model.block4:
    hooks.append(AttentionMap(m.attn.matmul))

sim_list = dict(
        [(x, []) for x in range(len(hooks))]
            # [(0, []),
            #  (1, []),
            #  (2, []),
            #  (3, []),
            #  (4, []),
            #  (5, []),
            #  (6, []),
            #  (7, []),
            #  (8, []),
            #  (9, []),
            #  (10, []),
            #  (11, []),
            #  ]
        )

val_loader = tqdm(val_loader, file=sys.stdout)
with torch.no_grad():
    for step, data in enumerate(val_loader):
        # print("step = ", step)
        images, labels = data
        # size = images.shape(0)
        images = images.cuda()
        labels = labels.cuda()
        pred_final = model(images)

        for itr_hook in range(len(hooks)):
            # if itr_hook != 0:
            #     continue
            # Hook attention
            attention = hooks[itr_hook].feature
            # sims = tokens_similarity(attention)
            attention_p = attention.mean(dim=1)

            sims = tokens_similarity(attention_p)
            # [:, 1:, :]
            # sims = torch.cosine_similarity(attention_p.unsqueeze(1), attention_p.unsqueeze(2), dim=3)
            # print(torch.mean(sims))
            # sims_item = torch.mean(sims, dim=(1,2))
            # sim_list.extend(sims_item.data.cpu().numpy())
            sim_list[itr_hook].extend(sims)
            print("idx = ", itr_hook, " sim = ", 1-np.mean(sim_list[itr_hook]))
            # print("idx = ", itr_hook, " sim = ", torch.mean(sims))
        pass
    pass
