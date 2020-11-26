include("../backbones/resnet.jl")
include("../backbones/fpn.jl")
include("../backbones/ssh.jl")
include("../../configs.jl")

"""
Takes final Context Head Outputs and converts these into proposals.
Same structure for BBox, Classifier and Landmark tasks.
task_len is 2 for classification, 4 for bbox and 10 for landmark
"""
struct HeadGetter layers; task_len; end

function HeadGetter(input_dim, num_anchors, task_len; scale_cnt=5, dtype=Array{Float64})
    layers = []
    for s in 1:scale_cnt
        push!(layers, Conv2D(1, 1, input_dim, num_anchors*task_len, dtype=dtype, bias=false))
    end
    return HeadGetter(layers, task_len)
end

function (hg::HeadGetter)(xs; train=true)
    proposals = []
    for (i, x) in enumerate(xs)
        proposal = hg.layers[i](x, train=train)
        batch_size = size(proposal)[end]
        num_proposals = Int(floor(prod(size(proposal)) / (hg.task_len * batch_size)))
        proposal = reshape(proposal, (batch_size, num_proposals, hg.task_len))
        push!(proposals, proposal)
    end
    if hg.task_len == 2 softmax(cat(proposals..., dims=2), dims=3)
    else return cat(proposals..., dims=2)
    end
end

"""
Our actual model that predicts bounding boxes and landmarks.
"""
struct RetinaFace
    backbone
    fpn
    head_module1
    head_module2
    class_conv1
    class_conv2
    bbox_conv1
    bbox_conv2
    landmark_conv1
    landmark_conv2
end

function RetinaFace(;dtype=Array{Float64}) 
    
    return RetinaFace(
        ResNet50(include_top=false, dtype=dtype), 
        FPN(dtype=dtype), SSH(dtype=dtype), SSH(dtype=dtype),
        HeadGetter(256, num_anchors, 2, dtype=dtype), HeadGetter(256, num_anchors, 2, dtype=dtype),
        HeadGetter(256, num_anchors, 4, dtype=dtype), HeadGetter(256, num_anchors, 4, dtype=dtype),
        HeadGetter(256, num_anchors, 10, dtype=dtype), HeadGetter(256, num_anchors, 10, dtype=dtype),
    )   
end

function (model::RetinaFace)(x; train=true)
    # first processes
    c2, c3, c4, c5 = model.backbone(x, return_intermediate=true, train=false)
    p_vals = model.fpn([c2, c3, c4, c5], train=train)
    
    # context head results will be kept
    ssh_vals1 = []
    ssh_vals2 = []
    # loss params
    loss1 = 0
    loss2 = 0
    
    # 1st context head module
    for p in p_vals push!(ssh_vals1, model.head_module1(p, train=train)) end
    if train
        class_vals1 = model.class_conv1(p_vals)
        bbox_vals1 = model.bbox_conv1(p_vals)
        landmark_vals1 = model.landmark_conv1(p_vals)
        # TODO: calculate loss
    end
    # 2nd context head module
    for s in ssh_vals1 push!(ssh_vals2, model.head_module2(s, train=train)) end
    class_vals2 = model.class_conv2(p_vals)
    bbox_vals2 = model.bbox_conv2(p_vals)
    landmark_vals2 = model.landmark_conv2(p_vals)
    # TODO: calculate loss
    
    return class_vals2, bbox_vals2, landmark_vals2
end