using Images, Knet

"""
Global hyper parameters are defined here for simplicity in
changing their values.
"""
# Image and Data Paths
wf_path         = "/datasets/widerface/WIDER_"
wf_labels_path  = "/datasets/widerface/retinaface/"
fddb_path       = "/datasets/fddb/"
aflw_path       = "/datasets/aflw/"

# Weight Paths
r50_mat_dir     = "./weights/imagenet-resnet-50-dag.mat"
save_dir        = "./weights/"
# load_path       = "./weights/model_6_6430.jld2"
load_path       = nothing

# Extra Paths
log_dir         = "./log_msgs.txt"

# Image Manipulation Metrics
crop_ratios     = [0.3, 0.45, 0.6, 0.8, 1.0]
img_size        = 640
avg_img         = channelview(Images.imresize(load("./data/avgImg.jpg"), (img_size, img_size)))

# Anchor Metrics
num_anchors     = 3
anchor_info     = [ 
    Dict([("stride", 4), ("anchors", [16, 20.16, 25.40])]),
    Dict([("stride", 8), ("anchors", [32, 40.32, 50.80])]),
    Dict([("stride", 16), ("anchors", [64, 80.63, 101.59])]),
    Dict([("stride", 32), ("anchors", [128, 161.26, 203.19])]),
    Dict([("stride", 64), ("anchors", [256, 322.54, 406.37])])
]

# Loss Calculation Metrics
scale           = 2^(1/3)
head1_pos_iou   = 0.7
head1_neg_iou   = 0.3
head2_pos_iou   = 0.5
head2_neg_iou   = 0.4
ohem_ratio      = 3
ohem_neg_iou    = 0.1
lambda1         = 1
lambda2         = 1

# Training Parameters
batch_size      = 10
num_epochs      = 80
weight_decay    = 0.00005
momentum        = 0.9
lrs             = [1e-3, 1e-2, 1e-3, 1e-4]
lr_change_epoch = [5, 55, 68]
mode            = 0 # 1 means first context head, 2 means second context head, 0 means no context head
start_epoch     = 1

# Testing Parameters
conf_level      = 0.9
nms_threshold   = 0.4

# Device Parameters
run_gpu = true
atype = run_gpu ? KnetArray{Float32} : Array{Float32}
