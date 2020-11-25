include("../backbones/resnet.jl")
include("../backbones/fpn.jl")
include("../backbones/ssh.jl")
include("../../configs.jl")

struct RetinaFace
    backbone
    fpn
    head_module1
    head_module2
    class_conv
    bbox_conv
    landmark_conv
end

function RetinaFace(;dtype=Array{Float64}) 
    return RetinaFace(
        ResNet50(include_top=false, dtype=dtype), 
        FPN(dtype=dtype), SSH(dtype=dtype), SSH(dtype=dtype),
        Conv2D(1, 1, 256, num_anchors*2, dtype=dtype, bias=false),
        Conv2D(1, 1, 256, num_anchors*4, dtype=dtype, bias=false),
        Conv2D(1, 1, 256, num_anchors*10, dtype=dtype, bias=false)
    )   
end

function (model::RetinaFace)(x; train=true)
    # first processes
    c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=false)
    p_vals = model.fpn([c2, c3, c4, c5], train=train)
    
    # context head results will be kept
    ssh_vals1 = []
    ssh_vals2 = []
    # classification, bbox and landmark resuls will be kept
    class_vals1, bbox_vals1, landmark_vals1 = [], [], []
    class_vals2, bbox_vals2, landmark_vals2 = [], [], []
    # loss params
    loss1 = 0
    loss2 = 0
    
    # 1st context head module
    for p in p_vals push!(ssh_vals1, model.head_module1(p, train=train)) end
    if train
        class_vals1 = [permutedims(model.class_conv(val, train=train), (4, 1, 2, 3)) for val in ssh_vals1]
        bbox_vals1 = [permutedims(model.bbox_conv(val, train=train), (4, 1, 2, 3)) for val in ssh_vals1]
        landmark_vals1 = [permutedims(model.landmark_conv(val, train=train), (4, 1, 2, 3)) for val in ssh_vals1]
    end
    # 2nd context head module
    for s in ssh_vals1 push!(ssh_vals2, model.head_module2(s, train=train)) end
    class_vals2 = [permutedims(model.class_conv(val, train=train), (4, 1, 2, 3)) for val in ssh_vals2]
    bbox_vals2 = [permutedims(model.bbox_conv(val, train=train), (4, 1, 2, 3)) for val in ssh_vals2]
    landmark_vals2 = [permutedims(model.landmark_conv(val, train=train), (4, 1, 2, 3)) for val in ssh_vals2]
    
    return class_vals2, bbox_vals2, landmark_vals2
end