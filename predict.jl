include("BBTNet/models/retinaface.jl")
include("./BBTNet/utils/draw.jl")
include("configs.jl")

val_data = WIDER_Data(wf_path * "val/", wf_labels_path * "val/", train=true, batch_size=2, dtype=atype)
(imgs, boxes), state = iterate(val_data)

model = RetinaFace(dtype=atype)
c, b, l = model(imgs, nothing, mode, false, 0)

# image conversion to actual shape and cpu type for the first image in the prediction
# img_cpu = Array(permutedims(imgs[:,:,:,1], (3,2,1)))
# combined_pts = permutedims(Array(cat(b[1], l[1], dims=2)), (2, 1))

# draw_boxes_and_landmarks(img_cpu, combined_pts)