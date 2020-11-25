include("../backbones/resnet.jl")
include("../backbones/fpn.jl")
include("../backbones/ssh.jl")

struct RetinaFace
    backbone
    fpn
    head_module1
    head_module2
end

function RetinaFace(;dtype=Array{Float64}) 
    return RetinaFace(
        ResNet50(include_top=false, dtype=dtype), 
        FPN(dtype=dtype), SSH(dtype=dtype), SSH(dtype=dtype)
    )   
end

function (model::RetinaFace)(x; train=true)
    c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=false)
    p_vals = model.fpn([c2, c3, c4, c5], train=train)
    ssh_vals = []
    for p in p_vals push!(ssh_vals, model.head_module1(p, train=train)) end
    ssh_vals2 = []
    for s in ssh_vals push!(ssh_vals2, model.head_module2(s, train=train)) end
    return ssh_vals2
end