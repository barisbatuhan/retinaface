include("../backbones/resnet.jl")

mutable struct RetinaFace
    resnet50
    atype
    weights_file::String
end

function RetinaFace(atype=Array{Float64})
    resnet50 = ResNet50(
        include_top=false, 
        atype=atype, 
        return_intermediate=true)
    return RetinaFace(resnet50, atype, "-")   
end

# mode, 0=>train, 1=>test
function (model::RetinaFace)(x, mode=1)
    c2, c3, c4, c5 = model.resnet50(x)
    p6 = conv4(convert(model.atype, xavier(3, 3, size(c5)[3], 256)), c5, stride=2, padding=1)
    p5 = conv4(convert(model.atype, xavier(1, 1, size(c5)[3], 256)), c5)
    p4 = _lateral_conn(model, c4, p5)
    p3 = _lateral_conn(model, c3, p4)
    p2 = _lateral_conn(model, c2, p3)
    return p2, p3, p4, p5, p6
end

function _lateral_conn(model::RetinaFace, left, up)
    up_sampled = unpool(up)
    resized_left = conv4(convert(model.atype, xavier(1, 1, size(left)[3], 256)), left)
    return up_sampled .+ resized_left
end