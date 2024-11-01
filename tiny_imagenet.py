# import torch
# _base_ = '../_base_/datasets/imagenet_bs64_swin_224.py'
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='NumpyToPIL', to_rgb=True),       # from BGR in cv2 to RGB  in PIL
#     dict(
#         type='torchvision/RandomResizedCrop',
#         size=size,
#         interpolation='bilinear'),            # accept str format interpolation mode
#     dict(type='torchvision/RandomHorizontalFlip', p=0.5),
#     # dict(
#     #     type='torchvision/TrivialAugmentWide',
#     #     interpolation='bilinear'),
#     dict(type='torchvision/PILToTensor'),
#
#     dict(type='torchvision/ConvertImageDtype', dtype="float"),
#     dict(
#         type='torchvision/Normalize',
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     ),
#     dict(type='torchvision/RandomErasing', p=0.1),
#     dict(type='PackInputs'),
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='NumpyToPIL', to_rgb=True),
#     dict(
#         type='torchvision/Resize',
#         size=size,
#         interpolation='bilinear'),
#     dict(type='torchvision/PILToTensor'),
#     dict(
#         type='torchvision/Normalize',
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225),
#     ),
#     dict(type='PackInputs'),
#
# ]

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type="CustomDataset",
        data_root = "",
        data_prefix='train',

        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=5,
    dataset=dict(
        type="CustomDataset",
        data_root="",
        data_prefix='val',

        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
