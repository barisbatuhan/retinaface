include("BBTNet/models/retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("./BBTNet/utils/draw.jl")

atype = KnetArray{Float32}
# atype = Array{Float32}
images_folder_dir = "../Datasets/WIDERFACE/WIDER_"

val_data = WIDER_Data(images_folder_dir * "val/", train=false, batch_size=2)
(imgs, boxes), state = iterate(val_data)
print("Got the first batch: ", size(imgs), "\n")

model = RetinaFace(dtype=atype)
imgs_permuted = convert(atype, permutedims(imgs, (3,2,1,4)))
c, b, l = model(imgs_permuted, train=false)

# prints first image in the batch
combined = cat(b[1], l[1], dims=2)
draw_boxes_and_landmarks(imgs[:,:,:,1], combined)