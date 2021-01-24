using Knet

"""
Global hyper parameters are defined here for simplicity in
changing their values.
"""
# Image and Data Paths
wf_path         = "/datasets/widerface/WIDER_"
wf_labels_path  = "/datasets/widerface/retinaface/"
wf_eval_path    = "./data/"
fddb_path       = "/datasets/fddb/"
aflw_path       = "/datasets/aflw/"

# Weight Paths
r50_mat_dir     = "./weights/imagenet-resnet-50-dag.mat"

# Image Manipulation Metrics
avg_img         = (123, 117, 104)

# Training Parameters
batch_size      = 10
num_epochs      = 27
weight_decay    = 0.0005
momentum        = 0.9
lrs             = [1e-3, 1e-2, 1e-3, 1e-4]
lr_change_epoch = [3, 17, 23]

# Evaluation Parameters
mAP_ious        = [0.5, 0.05, 0.95] # start, step size, end

# Device Parameters
run_gpu = true
atype = run_gpu ? KnetArray{Float32} : Array{Float32}

# Anchor Metrics
lat_3_anchors   = [ 
    Dict([("stride", 8), ("anchors", [16, 32])]),
    Dict([("stride", 16), ("anchors", [64, 128])]),
    Dict([("stride", 32), ("anchors", [256, 512])])
]

lat_5_anchors = [ 
    Dict([("stride", 4), ("anchors", [16, 20.16, 25.40])]),
    Dict([("stride", 8), ("anchors", [32, 40.32, 50.80])]),
    Dict([("stride", 16), ("anchors", [64, 80.63, 101.59])]),
    Dict([("stride", 32), ("anchors", [128, 161.26, 203.19])]),
    Dict([("stride", 64), ("anchors", [256, 322.54, 406.37])])
]

# Loss Calculation Metrics
head1_pos_iou   = 0.7
head1_neg_iou   = 0.3
head2_pos_iou   = 0.5
head2_neg_iou   = 0.4 # 0.3 # --> for no cascaded structure
ohem_ratio      = 3 
lambda1         = 2
lambda2         = 1



