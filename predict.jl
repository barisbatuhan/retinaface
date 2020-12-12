include("BBTNet/models/retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("configs.jl")

images_folder_dir = "../Datasets/WIDERFACE/WIDER_"

val_data = WIDER_Data(images_folder_dir * "val/", train=false, batch_size=2, dtype=atype)
(imgs, boxes), state = iterate(val_data)
print("Got the first batch: ", size(imgs), "\n")

model = RetinaFace(dtype=atype)
c, b, l = model(imgs, train=false)

# include("./BBTNet/utils/draw.jl")
# combined = permutedims(cat(b[1], l[1], dims=2), (2, 1))
# image = draw_boxes_and_landmarks(imgs[:,:,:,1], combined)
# imshow(image)