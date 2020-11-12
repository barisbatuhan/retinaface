include("./BBTNet/datasets/WIDERFACE.jl")
include("./BBTNet/utils/draw.jl")
include("./BBTNet/layers/conv2d_block.jl")

root_dir = "../Datasets/WIDERFACE/WIDER_"
train_data = WIDER_Data(root_dir * "train/")

(imgs, boxes), state = iterate(train_data)
print("Got the first batch: ", size(imgs))

# For visualizing ground truths
# marked_img = draw_boxes_and_landmarks(imgs[:,:,:,5], boxes[5])
# imshow(marked_img)

# For predicting purposes
model = Conv2D_Block(3, 2, [(3, 3)], [32], paddings=[1])
out = model(permutedims(imgs, [3,2,1,4]))
print("An output is created with the size: ", size(out), "\n")