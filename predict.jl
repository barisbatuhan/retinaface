include("BBTNet/models/retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("configs.jl")

data = WIDER_Data(wf_path * "val/", wf_labels_path * "val/", train=true, batch_size=1, dtype=atype)
(imgs, boxes), state = iterate(val_data)
print("Got the first batch: ", size(imgs), "\n")

model = RetinaFace(dtype=atype)
c, b, l = model(imgs, train=false)

# include("./BBTNet/utils/draw.jl")
# combined = permutedims(cat(b[1], l[1], dims=2), (2, 1))
# image = draw_boxes_and_landmarks(imgs[:,:,:,1], combined)
# imshow(image)