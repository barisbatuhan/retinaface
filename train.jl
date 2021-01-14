include("BBTNet/models/retinaface.jl")
include("BBTNet/models/pth_load_retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("configs.jl")

# import Random
# Random.seed!(42)

train_dir = wf_path * "train/"
labels_dir = wf_labels_path * "train/"
data = WIDER_Data(train_dir, labels_dir, train=false, shuffle=false, batch_size=batch_size, dtype=atype)

print(data.files, "\n")

print("[INFO] Data is loaded!\n")

model = nothing
if load_path === nothing
    model = RetinaFace(dtype=atype)
else
    model = load_model(load_path)
end

# model = RetinaFace(dtype=atype)
# model = load_pth_model(model, save_dir * "Resnet50.jld"; dtype=atype, load_heads=false)
print("[INFO] Model is loaded!\n")

train_model(model, data, save_dir=save_dir)
