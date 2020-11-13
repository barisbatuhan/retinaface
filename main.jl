include("./BBTNet/datasets/WIDERFACE.jl")
include("./BBTNet/utils/draw.jl")
include("./BBTNet/layers/conv2d_block.jl")
include("./BBTNet/backbones/resnet.jl")

root_dir = "../Datasets/WIDERFACE/WIDER_"
train_data = WIDER_Data(root_dir * "train/")

(imgs, boxes), state = iterate(train_data)
print("Got the first batch: ", size(imgs),'\n')

# For visualizing ground truths
# marked_img = draw_boxes_and_landmarks(imgs[:,:,:,5], boxes[5])
# imshow(marked_img)

# For predicting purposes
# model = Conv2D_Block(3, 2, [(3, 3)], [32], paddings=[1])
# out = model(permutedims(imgs, [3,2,1,4]))
# print("An output is created with the size: ", size(out), "\n")

# For predicting with ResNet50
model = ResNet50(include_top=false, return_intermediate=true)
c2, c3, c4, c5 = model(permutedims(imgs, [3,2,1,4]))
print(size(c2), size(c3), size(c4), size(c5))