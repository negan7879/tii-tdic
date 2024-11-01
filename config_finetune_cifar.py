_base_ = [
    '../_base_/models/vit-base-p16.py',
    '../_base_/datasets/cifar100_bs16.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]
# find_unused_parameters=True

num_classes = 100
data_preprocessor = dict(
    num_classes = num_classes
)
# optim_wrapper = dict(
#
# )
teacher_path = None
model_list = ["ViT_B", "ViT_S", "ViT_T", "DeiT-B","DeiT-S","DeiT-T",
              "LeViT_192", "LeViT_256", "LeViT_128",
              "EViT_Base", "EViT_Tiny","EViT_Small",
              "DyViT_B","DyViT_S","DyViT_T",
              "T2T_ViT_14","T2T_ViT_19",
              "PVT_Small","PVT_Tiny", "PVT_Medium"]

model_name = "EViT_Small"

assert model_name in model_list
# six_card new
server = "six_card"
model_type = "student"
if server == "411":
    if model_type == "student":
        pretrained_dir = "/datb/notebook/CPS/work1/mmpretrain/work_dirs/config_distiller_finetune/20230916_170242/epoch_426.pth"
    elif model_type == "teacher":
        pretrained_dir = "/datb/notebook/CPS/work1/attndistill/dino_deitsmall8_pretrain.pth"
    data_root = "/datb/notebook/CPS/work1/data/tiny-imagenet-200"
elif server == "old":
    # pretrained_dir = "/media/data1/zhangzherui/code/attndistill/dino_deitsmall8_pretrain.pth"
    pretrained_dir = "/media/data1/zhangzherui/code/mmpretrain/work_dirs/distiller/best_val_loss_epoch_796.pth"
    # data_root = "/media/data2/zhangzherui/data/tiny-imagenet-200"
    data_root = "/media/data1/zhangzherui/data/imagenet"
elif server == "new":
    data_root = "/media/data/zhangzherui/data/cifar100"


    # *************************************** vit ************************
    # base
    # pretrained_dir = "/media/data/zhangzherui/data/weights/deit_base_patch16_224-b5f2ef4d.pth"
    # small
    # pretrained_dir = "/media/data/zhangzherui/data/weights/deit_small_patch16_224-cd65a155.pth"
    # tiny
    # pretrained_dir = "/media/data/zhangzherui/data/weights/deit_tiny_patch16_224-a1311bcf.pth"

    # *************************************** DeiT ************************
    if model_name == "DeiT-B":
        find_unused_parameters = True  # for deit
        teacher_path = "/media/data/zhangzherui/data/weights/vit_base_finetune_tinyimagenet.pth"
        pretrained_dir = "/media/data/zhangzherui/data/weights/deit_base_distilled_patch16_224-df68dfff.pth"
    elif model_name == "DeiT-S":
        teacher_path = "/media/data/zhangzherui/data/weights/vit_base_finetune_tinyimagenet.pth"
        find_unused_parameters = True  # for deit
        pretrained_dir = "/media/data/zhangzherui/data/weights/deit_small_distilled_patch16_224-649709d9.pth"
    elif model_name == "DeiT-T":
        teacher_path = "/media/data/zhangzherui/data/weights/vit_base_finetune_tinyimagenet.pth"
        find_unused_parameters = True  # for deit
        pretrained_dir = "/media/data/zhangzherui/data/weights/deit_tiny_distilled_patch16_224-b40b3cf7.pth"
    elif model_name == "LeViT_192":
        teacher_path = "/media/data/zhangzherui/data/weights/vit_base_finetune_tinyimagenet.pth"
        find_unused_parameters = True  # for deit
        pretrained_dir = "/media/data/zhangzherui/data/weights/LeViT-192-92712e41.pth"
    elif model_name == "LeViT_256":
        teacher_path = "/media/data/zhangzherui/data/weights/vit_base_finetune_tinyimagenet.pth"
        find_unused_parameters = True  # for deit
        pretrained_dir = "/media/data/zhangzherui/data/weights/LeViT-256-13b5763e.pth"
    elif model_name == "EViT_Base":
        pretrained_dir = "/media/data/zhangzherui/data/weights/deit_base_patch16_224-b5f2ef4d.pth"
    elif model_name == "EViT_Tiny":
        pretrained_dir = "/media/data/zhangzherui/data/weights/deit_tiny_patch16_224-a1311bcf.pth"
    elif model_name == "T2T_ViT_19":
        pretrained_dir = "/media/data/zhangzherui/data/weights/82.4_T2T_ViTt_19.pth.tar"
    elif model_name == "PVT_Tiny":
        pretrained_dir = "/media/data/zhangzherui/data/weights/pvt_tiny.pth"
    # PVT Small

    # pretrained_dir = "/media/data/zhangzherui/data/weights/pvt_small.pth"
    # pretrained_dir = "/media/data/zhangzherui/data/weights/pvt_medium.pth"
    # pretrained_dir = "/media/data/zhangzherui/data/weights/pvt_tiny.pth"


elif server == "six_card":
    # find_unused_parameters = True  # for deit + levit
    # data_root = "/work/zhangzherui/data/tiny-imagenet-200"
    data_root = "/work/zhangzherui/data/cifar100"
    # pretrained_dir = "/work/zhangzherui/data/weights/deit_base_patch16_224-b5f2ef4d.pth"
    if model_name == "ViT_S":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_small_patch16_224-cd65a155.pth"
    elif model_name == "ViT_B":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_base_patch16_224-b5f2ef4d.pth"
    elif model_name == "ViT_T":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_tiny_patch16_224-a1311bcf.pth"
    elif model_name == "DyViT_B":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_base_patch16_224-b5f2ef4d.pth"
    elif model_name == "DyViT_S":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_small_patch16_224-cd65a155.pth"
    elif model_name == "DyViT_T":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_tiny_patch16_224-a1311bcf.pth"
    elif model_name == "EViT_Small":
        pretrained_dir = "/work/zhangzherui/data/weights/deit_small_patch16_224-cd65a155.pth"
    elif model_name == "T2T_ViT_14":
        pretrained_dir = "/work/zhangzherui/data/weights/81.7_T2T_ViTt_14.pth.tar"
    elif model_name == "PVT_Small":
        pretrained_dir = "/work/zhangzherui/data/weights/pvt_small.pth"
    elif model_name == "PVT_Medium":
        pretrained_dir = "/work/zhangzherui/data/weights/pvt_medium.pth"
    elif model_name == "LeViT_192":
        pretrained_dir = '/work/zhangzherui/data/weights/LeViT-192-92712e41.pth'
        find_unused_parameters = True  # for deit
        teacher_path = "/work/zhangzherui/code/mmpretrain/work_dirs/config_finetune_cifar/weights/vit_s.pth"
    elif model_name == "LeViT_256":
        pretrained_dir = '/work/zhangzherui/data/weights/LeViT-256-13b5763e.pth'
        find_unused_parameters = True  # for deit
        teacher_path = "/work/zhangzherui/code/mmpretrain/work_dirs/config_finetune_cifar/weights/vit_s.pth"
    elif model_name == "LeViT_128":
        pretrained_dir = '/work/zhangzherui/data/weights/LeViT-128-b88c2750.pth'
        # find_unused_parameters = True  # for deit
        # teacher_path = "/work/zhangzherui/code/mmpretrain/work_dirs/config_finetune_cifar/weights/vit_s.pth"
            # find_unused_parameters = True  # for deit
    # t2t 14
    # pretrained_dir = '/work/zhangzherui/data/weights/81.7_T2T_ViTt_14.pth.tar'
    # t2t 19
    # pretrained_dir = '/work/zhangzherui/data/weights/82.4_T2T_ViTt_19.pth.tar'
    # t2t 24
    # pretrained_dir = '/work/zhangzherui/data/weights/82.6_T2T_ViTt_24.pth.tar'

    # levit
    # pretrained_dir = '/work/zhangzherui/data/weights/LeViT-128-b88c2750.pth'
    # pretrained_dir = '/work/zhangzherui/data/weights/LeViT-192-92712e41.pth'


model = dict(
    _delete_ = True,
    type='ClassificationFintune',
    pretrained_dir = pretrained_dir,
    need_teacher = teacher_path != None,
    teacher_path = teacher_path,
    model_name = model_name,
    num_classes = num_classes,
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)
batch_size = 64
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset = dict(
        data_root = data_root
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    dataset = dict(
        data_root = data_root
    )
)
# val_evaluator = dict(
#     _delete_ = True,
#     type='VAL_LOSS_Metric',
# )

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = val_dataloader
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint = dict(
        interval = 3,
        save_best="auto",
        # rule='less', # 越大越好,
        max_keep_ckpts=2
        #https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html#checkpointhook
    )
)
train_cfg = dict(max_epochs = 100)
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=False),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-6, by_epoch=True, begin=20)
]


bb_lr_mult = 0.01
base_lr = 5e-5

optim_wrapper = dict(
optimizer=dict(lr=base_lr * 1024 / 512),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,

        lr_mult = bb_lr_mult,
        custom_keys={
            'head.weight' : dict(lr_mult=1.0, ),
            'head.bias' : dict(lr_mult=1.0, ),
            'head_dist' : dict(lr_mult=1.0, ),
        }
        #     # 'patch_embed' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.0' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.1' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.2' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.3' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.4' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.5' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.6' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.7' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.8' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.9' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # 'blocks.10' : dict(lr_mult=bb_lr_mult, decay_mult=0),
        #     # '.cls_token': dict(decay_mult=0.0),
        #     # '.pos_embed': dict(decay_mult=0.0)
        # }
        ),
    clip_grad=dict(max_norm=5.0),
)

auto_scale_lr = dict(base_batch_size=64)

visualizer=dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

