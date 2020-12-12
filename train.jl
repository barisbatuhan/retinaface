include("BBTNet/models/retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("configs.jl")


images_folder_dir = "../Datasets/WIDERFACE/WIDER_"

data = WIDER_Data(images_folder_dir * "train/", train=true, batch_size=1, dtype=atype)

model = RetinaFace(dtype=atype)
train_model(model, data)