import os

_base_ = [
    '../_base_/models/vit-base-p16.py',
    './tiny_imagenet.py',
    # '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
# broadcast_buffers=False
find_unused_parameters=True
num_classes = 200
data_preprocessor = dict(
    num_classes = num_classes
)

server = "six_card"
if server == "411":
    pretrained_dir = "/datb/notebook/Anonymous/work1/attndistill/dino_deitsmall8_pretrain.pth"
    data_root = "/datb/notebook/Anonymous/work1/data/tiny-imagenet-200"
elif server == "old":
    pretrained_dir = "/media/data1/Anonymous/code/mmpretrain/model_latest.pth"
    data_root = "/media/data1/Anonymous/data/imagenet"
elif server == "new":
    data_root = "/media/data/Anonymous/data/tiny-imagenet-200"
    # data_root = "/media/data/Anonymous/data/imagenet"
    pretrained_dir = "/media/data/Anonymous/code/VIT_DIST/weights/tinyimagenet/VIT_BASE/1.0/model_latest.pth"
    # pretrained_dir = "/media/data/Anonymous/code/mmpretrain/deit-tiny.pth"
elif server == "six_card":
    data_root = "/work/Anonymous/data/tiny-imagenet-200"
    # pretrained_dir = "/work/Anonymous/data/model_latest.pth"
    # small
    pretrained_dir = "/work/Anonymous/data/weights/vit_small_finetune_tinyimagenet.pth"


batch_size = 64
model = dict(
    _delete_ = True,
    type='ClassificationDiffRateViT',
    pretrained_dir = pretrained_dir,
    is_plot = True,
    num_classes = num_classes,
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)

train_dataloader = dict(
    batch_size=batch_size,
    dataset = dict(
        data_root = data_root,
        # split='train',

        # indices=5000,
    )
)
val_dataloader = dict(
    batch_size=batch_size,

    dataset = dict(
        data_root = data_root,
        # split='val',

    )
)
# val_evaluator = dict(
#     _delete_ = True,
#     type='VAL_LOSS_Metric',
# )
val_evaluator = dict(
    _delete_ = True,
    type='Accuracy',
    topk=(1, 5)
    # type='Muti_Acc_Metric',
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint = dict(
        interval = 1,
        save_best="auto",
        max_keep_ckpts=2,
        # out_dir = output_dir
        #https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html#checkpointhook
    ),
    # logger = dict(
    #     out_dir = log_dir
    # )

)
my_epoch = 150
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


bb_lr_mult = 1.0
base_lr = 5e-5

optim_wrapper = dict(
optimizer=dict(lr=base_lr * 1024 / 512),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            'selected_probability' : dict(lr_mult=bb_lr_mult, decay_mult=0),
            # 'pos_embed' : dict(lr_mult=bb_lr_mult, decay_mult=0),
            # 'patch_embed' : dict(lr_mult=bb_lr_mult, decay_mult=0),

            # '.cls_token': dict(decay_mult=0.0),
            # '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

auto_scale_lr = dict(base_batch_size=64)

visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

