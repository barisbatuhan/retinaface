"""
Global hyper parameters are defined here for simplicity in
changing their values.
"""
# Anchor Metrics
num_anchors = 3

# Loss Calculation Metrics
scale           = 2^(1/3)
head1_pos_iou   = 0.7
head1_neg_iou   = 0.3
head2_pos_iou   = 0.5
head2_neg_iou   = 0.4

# Training Parameters
batch_size      = 24
epoch           = 100
decay1          = 70
decay2          = 9