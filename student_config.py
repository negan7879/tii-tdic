
patch_size=16
embed_dim=384
depth=12
num_heads=6
mlp_ratio=4,
qkv_bias=True
teacher_embed_dim=384
model = dict(
    type='stu_VisionTransformer',
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    qkv_bias=qkv_bias,
    teacher_embed_dim = teacher_embed_dim
)