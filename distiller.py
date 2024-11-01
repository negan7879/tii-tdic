_base_ = [
    '../_base_/models/vit-base-p16.py',
    './tiny_imagenet.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

find_unused_parameters=True

# optim_wrapper = dict(
#
# )
server = "old"
if server == "411":
    teacher_pretrained = "/datb/notebook/CPS/work1/attndistill/dino_deitsmall8_pretrain.pth"
    data_root = "/datb/notebook/CPS/work1/data/tiny-imagenet-200"
elif server == "old":
    teacher_pretrained = "/media/data1/zhangzherui/code/attndistill/dino_deitsmall8_pretrain.pth"
    data_root = "/media/data2/zhangzherui/data/tiny-imagenet-200"
elif server == "new":
    teacher_pretrained = "/media/data/zhangzherui/code/mmpretrain/dino_deitsmall8_pretrain.pth"
    data_root = "/media/data/zhangzherui/data/tiny-imagenet-200"
model = dict(
    _delete_ = True,
    type='ClassificationDistiller',
    teacher_pretrained = teacher_pretrained,
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
batch_size = 64
train_dataloader = dict(
    batch_size=batch_size,
    dataset = dict(
        data_root = data_root
    )
)
val_dataloader = dict(
    batch_size=batch_size,

    dataset = dict(
        data_root = data_root
    )
)
val_evaluator = dict(
    _delete_ = True,
    type='VAL_LOSS_Metric',
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint = dict(
        interval = 3,
        save_best="val_loss",
        rule='less', # 越大越好,
        max_keep_ckpts=2
        #https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html#checkpointhook
    )
)
my_epoch = 800
my_warmup_epoch = 20
train_cfg = dict(max_epochs = my_epoch)
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=my_warmup_epoch,
        # update by iter
        convert_to_iter_based=False),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=my_warmup_epoch, T_max=my_epoch-my_warmup_epoch)
]

# base_lr = 5e-5
base_lr = 5e-4
optim_wrapper = dict(
optimizer=dict(lr=base_lr * 1024 / 512),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            'cls_token': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            # 'patch_embed' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.0' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.1' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.2' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.3' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.4' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.5' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.6' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.7' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.8' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.9' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.10' : dict(lr_mult=0.1, decay_mult=0),
            # 'blocks.11' : dict(lr_mult=0.1, decay_mult=0),
        }),
    clip_grad=dict(max_norm=5.0),
)


visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])
