# optimizer
max_lr=1e-4
optimizer = dict(type='AdamW', lr=max_lr, betas=(0.95, 0.99), weight_decay=0.0001,)
# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=max_lr,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=8000 * 24)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=4000)
evaluation = dict(by_epoch=False,
        start=0,
        interval=4000,
        pre_eval=True,
        rule='less',
        save_best='abs_rel',
        greater_keys=("a1", "a2", "a3"),
        less_keys=("abs_rel", "rmse"))
#runner = dict(type='EpochBasedRunner', max_epochs=24)
#checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=1)
#evaluation = dict(by_epoch=True, interval=1, pre_eval=True)
