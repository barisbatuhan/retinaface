"""
All of these structures are mostly taken from the GitHub repo:
https://github.com/denizyuret/Knet.jl/blob/master/examples/resnet/resnet.jl
"""

using Knet, ArgParse

include("../layers/conv2d_block.jl")
include(Knet.dir("data","imagenet.jl"))

struct ResNet50
    w
    ms
    include_top::Bool
    return_intermediate::Bool
end

function ResNet50(;
    weight_path::String="imagenet-resnet-50-dag", 
    atype=Array{Float64},
    include_top::Bool=true,
    return_intermediate::Bool=false)
    model = matconvnet(weight_path)
    w, ms = _get_params(model["params"], atype)
    return ResNet50(w, ms, include_top, return_intermediate)
end

# mode, 0=>train, 1=>test
function (model::ResNet50)(x; mode=1)
    # layer 1
    conv1  = conv4(model.w[1], x; padding=3, stride=2) .+ model.w[2]
    bn1    = batchnorm(model.w[3:4],conv1, model.ms; mode=mode)
    pool1  = pool(bn1; padding=1, window=3, stride=2)

    # layer 2,3,4,5
    r2 = _reslayerx5(model.w[5:34], pool1, model.ms; strides=[1,1,1,1], mode=mode)
    r3 = _reslayerx5(model.w[35:73], r2, model.ms; mode=mode)
    r4 = _reslayerx5(model.w[74:130], r3, model.ms; mode=mode) # 5
    r5 = _reslayerx5(model.w[131:160], r4, model.ms; mode=mode)

    # fully connected layer
    if model.include_top
        pool5  = pool(r5; stride=1, window=7, mode=2)
        fc1000 = model.w[161] * mat(pool5) .+ model.w[162]
        if model.return_intermediate
            return r2, r3, r4, r5, fc1000
        else
            return fc1000
        end
    end
    if model.return_intermediate
        return r2, r3, r4, r5
    else
        return r5
    end
end

function _reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function _reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu.(_reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end

function _reslayerx2(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = _reslayerx1(w[1:3],x,ms; padding=pads[1], stride=strides[1], mode=mode)
    bb = _reslayerx1(w[4:6],ba,ms; padding=pads[2], stride=strides[2], mode=mode)
    bc = _reslayerx0(w[7:9],bb,ms; padding=pads[3], stride=strides[3], mode=mode)
end

function _reslayerx3(w,x,ms; pads=[0,0,1,0], strides=[2,2,1,1], mode=1) # 12
    a = _reslayerx0(w[1:3],x,ms; stride=strides[1], padding=pads[1], mode=mode)
    b = _reslayerx2(w[4:12],x,ms; strides=strides[2:4], pads=pads[2:4], mode=mode)
    relu.(a .+ b)
end

function _reslayerx4(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu.(x .+ _reslayerx2(w,x,ms; pads=pads, strides=strides, mode=mode))
end

function _reslayerx5(w,x,ms; strides=[2,2,1,1], mode=1)
    x = _reslayerx3(w[1:12],x,ms; strides=strides, mode=mode)
    for k = 13:9:length(w)
        x = _reslayerx4(w[k:k+8],x,ms; mode=mode)
    end
    return x
end

function _get_params(params, atype)
    len = length(params["value"])
    ws, ms = [], []
    for k = 1:len
        name = params["name"][k]
        value = convert(Array{Float32}, params["value"][k])

        if endswith(name, "moments")
            push!(ms, reshape(value[:,1], (1,1,size(value,1),1)))
            push!(ms, reshape(value[:,2], (1,1,size(value,1),1)))
        elseif startswith(name, "bn")
            push!(ws, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "fc") && endswith(name, "filter")
            push!(ws, transpose(reshape(value,(size(value,3),size(value,4)))))
        elseif startswith(name, "conv") && endswith(name, "bias")
            push!(ws, reshape(value, (1,1,length(value),1)))
        else
            push!(ws, value)
        end
    end
    map(wi->convert(atype, wi), ws),
    map(mi->convert(atype, mi), ms)
end
