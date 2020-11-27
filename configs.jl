"""
Global hyper parameters are defined here for simplicity in
changing their values.
"""
# Image Crop Metrics
crop_ratios     = [0.3, 0.45, 0.6, 0.8, 1.0]
img_size        = 640

# Anchor Metrics
num_anchors     = 3
anchor_info     = [ 
    Dict([("stride", 4), ("anchors", [16, 20.16, 25.40])]),
    Dict([("stride", 8), ("anchors", [32, 40.32, 50.80])]),
    Dict([("stride", 16), ("anchors", [64, 80.63, 101.59])]),
    Dict([("stride", 32), ("anchors", [128, 161.26, 203.19])]),
    Dict([("stride", 64), ("anchors", [256, 322.54, 406.37])])
]
variances       = [0.1, 0.2]

# Loss Calculation Metrics
scale           = 2^(1/3)
head1_pos_iou   = 0.7
head1_neg_iou   = 0.3
head2_pos_iou   = 0.5
head2_neg_iou   = 0.4
ohem_ratio      = 3
lambda1         = 1
lambda2         = 1

# Training Parameters
batch_size      = 24
epoch           = 100
decay1          = 70
decay2          = 9

# Testing Parameters
conf_level      = 0.8