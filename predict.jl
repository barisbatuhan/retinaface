include("BBTNet/models/retinaface.jl")
include("./BBTNet/utils/draw.jl")
include("configs.jl")

val_data = WIDER_Data(wf_path * "val/", wf_labels_path * "val/", train=true, batch_size=2, dtype=atype)
(imgs, boxes), state = iterate(val_data)

model = RetinaFace(dtype=atype)
c, b, l = model(imgs, mode=mode, train=false)

print("Confidences: ", c, "\n")
print("Boxes: ", b, "\n")
print("Landmarks: ", l, "\n")