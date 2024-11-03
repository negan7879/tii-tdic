_base_ = [
    '../_base_/models/vit-base-p16.py',
    './tiny_imagenet.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]


model = dict(
    head=dict(
        num_classes=200, # 修改为200个类别
    )
)

train_dataloader = dict(
    batch_size=128,
    dataset = dict(
        data_root = "/datb/notebook/Anonymous/work1/data/tiny-imagenet-200"
    )
)
val_dataloader = dict(
    batch_size=128,

    dataset = dict(
        data_root = "/datb/notebook/Anonymous/work1/data/tiny-imagenet-200"
    )
)

default_hooks = dict(
    checkpoint = dict(
        interval = -1,
        save_best="auto",
        rule='greater', # 越大越好,
        max_keep_ckpts=2
        #https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html#checkpointhook
    )
)
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)



