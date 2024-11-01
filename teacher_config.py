
patch_size=8
embed_dim=384
depth=12
num_heads=6
mlp_ratio=4,
qkv_bias=True
model = dict(
    type='tea_VisionTransformer',
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    qkv_bias=qkv_bias
)