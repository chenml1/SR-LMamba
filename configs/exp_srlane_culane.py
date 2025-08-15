_base_ = [
    "./datasets/culane.py",
    "./models/srlane_r18.py"
]

work_dirs = "work_dirs/sr_cu"

iou_loss_weight = 2.
cls_loss_weight = 2.
l1_loss_weight = 0.2
angle_loss_weight = 15
cross_attn_loss_weight = 0. 05 
seg_loss_weight = 0. 5 

total_iter = 44440
batch_size = 40
eval_ep = 3
workers = 8
log_interval = 500

precision = "16-mixed"  # "32"

optimizer = dict(
    type="AdamW",
    lr=5e-4,                 
    betas=(0.9, 0.999),
    weight_decay=0.05,        
    eps=1e-8
)


scheduler = dict(
    type="warmup",          
    warm_up_iters=1200,       
    total_iters=total_iter,

    decay_type="cosine",      
    min_lr=5e-7            
)
data_loader = dict(
    prefetch_factor=4,       
    pin_memory=True,          
    persistent_workers=True, 
    shuffle_buffer_size=5000  
)

compute = dict(
    torch_compile=True,       
    cudnn_benchmark=True,     
    channels_last=True       
)


