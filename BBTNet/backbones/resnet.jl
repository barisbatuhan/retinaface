using MAT
using Printf

include("../core/layers.jl")
include("../core/blocks.jl")
include("../../configs.jl")

"""
Required ResNet structures are implemented there.
"""

mutable struct ResNet50 layer1; layer2; layer3; layer4; layer5; fc; dtype; classes; end

function ResNet50(;input_dim=3, dtype=Array{Float32}, pdrop=0, bias=false, include_top=true)
    layer1 = _InitLayer(input_dim=input_dim, dtype=dtype, pdrop=pdrop, bias=true)
    layer2 = Chain([
        Residual_1x3x1(64, [64, 64, 256], downsample=true, dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(256, [64, 64, 256], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(256, [64, 64, 256], dtype=dtype, pdrop=pdrop, bias=bias)
    ])
    layer3 = Chain([
        Residual_1x3x1(256, [128, 128, 512], downsample=true, dtype=dtype, pdrop=pdrop, 
                        ds_3x3_stride=2, bias=bias),
        Residual_1x3x1(512, [128, 128, 512], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(512, [128, 128, 512], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(512, [128, 128, 512], dtype=dtype, pdrop=pdrop, bias=bias)
    ])
    layer4 = Chain([
        Residual_1x3x1(512, [256, 256, 1024], downsample=true, dtype=dtype, pdrop=pdrop, 
                        ds_3x3_stride=2, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(1024, [256, 256, 1024], dtype=dtype, pdrop=pdrop, bias=bias)
    ])
    layer5 = Chain([
        Residual_1x3x1(1024, [512, 512, 2048], downsample=true, dtype=dtype, pdrop=pdrop, 
                        ds_3x3_stride=2, bias=bias),
        Residual_1x3x1(2048, [512, 512, 2048], dtype=dtype, pdrop=pdrop, bias=bias),
        Residual_1x3x1(2048, [512, 512, 2048], dtype=dtype, pdrop=pdrop, bias=bias),
    ])

    fc = nothing
    if include_top fc = Dense(2048, 1000) end

    return ResNet50(layer1, layer2, layer3, layer4, layer5, fc, dtype, nothing)
end

function (rn::ResNet50)(x; train=true, return_intermediate=true)
    c2 = rn.layer2(rn.layer1(x, train=true), train=true)
    c3 = rn.layer3(c2, train=true)
    c4 = rn.layer4(c3, train=true)
    c5 = rn.layer5(c4, train=true)
    if rn.fc === nothing
        if return_intermediate return c2, c3, c4, c5, nothing
        else return c5
        end
    else
        h, w, c, n = size(c5)
        # assuming image is square shaped
        p = pool(c5, window=h, mode=2)[1,1,:,:]
        res = rn.fc(p, train=true)
        if return_intermediate return c2, c3, c4, c5, res
        else return res
        end
    end
end

function evaluate_model(rn::ResNet50, x, top=5)
    res = rn(convert(rn.dtype, x), train=false, return_intermediate=false)
    z1 = vec(Array(res))
    s1 = sortperm(z1,rev=true)
    p1 = exp.(logp(z1))
    for ind in s1[1:top]
        print("$(rn.classes[ind]): $(@sprintf("%.2f",p1[ind]*100))%\n")
    end
end

struct _InitLayer conv_bn; end

function _InitLayer(;input_dim=3, output_dim=64, dtype=Array{Float32}, pdrop=0, bias=false)
    return _InitLayer(
        ConvBn(7, 7, input_dim, output_dim, bias=bias, padding=3, 
                stride=2, dtype=dtype, pdrop=pdrop)
    )
end

function (il::_InitLayer)(x; train=true)
    x_val = pool(relu.(il.conv_bn(x, train=train)), window=3, stride=2, padding=1)
    return x_val
end

# bn_mom, bn_b, bn_mult
function load_mat_weights(rn::ResNet50, path)
    conv_w, conv_b, fc_w, fc_b, bn_mom, bn_b, bn_mult, rn.classes = _get_params(path)
    
    # setting fc layer parameters
    rn.fc.w = convert(rn.dtype, fc_w)
    rn.fc.b = convert(rn.dtype, fc_b)
    
    idx = 1
    mom_idx = 1
    
    # setting layer 1
    rn.layer1.conv_bn.conv.w = convert(rn.dtype, conv_w[1])
    rn.layer1.conv_bn.conv.b = convert(rn.dtype, conv_b)
    rn.layer1.conv_bn.bn.bn_moments = bnmoments(mean=bn_mom[1], var=bn_mom[2])
    rn.layer1.conv_bn.bn.bn_params = vcat(vec(bn_mult[1]), vec(bn_b[1]))
    idx += 1; mom_idx += 2; 
    
    # setting layer 2
    rn.layer2.layers[1] = load_mat_weights(
        rn.layer2.layers[1], conv_w[idx:idx+3], bn_mom[mom_idx:mom_idx+7], bn_b[idx:idx+3], bn_mult[idx:idx+3]
    )
    idx += 4; mom_idx += 8; 

    for i in 2:size(rn.layer2.layers, 1)
        rn.layer2.layers[i] = load_mat_weights(
            rn.layer2.layers[i], conv_w[idx:idx+2], bn_mom[mom_idx:mom_idx+5], bn_b[idx:idx+2], bn_mult[idx:idx+2]
        )
        idx += 3; mom_idx += 6; 
    end
    
    
    # setting layer 3
    rn.layer3.layers[1] = load_mat_weights(
        rn.layer3.layers[1], conv_w[idx:idx+3], bn_mom[mom_idx:mom_idx+7], bn_b[idx:idx+3], bn_mult[idx:idx+3]
    )
    idx += 4; mom_idx += 8; 

    for i in 2:size(rn.layer3.layers, 1)
        rn.layer3.layers[i] = load_mat_weights(
            rn.layer3.layers[i], conv_w[idx:idx+2], bn_mom[mom_idx:mom_idx+5], bn_b[idx:idx+2], bn_mult[idx:idx+2]
        )
        idx += 3; mom_idx += 6; 
    end
    
    # setting layer 4
    rn.layer4.layers[1] = load_mat_weights(
        rn.layer4.layers[1], conv_w[idx:idx+3], bn_mom[mom_idx:mom_idx+7], bn_b[idx:idx+3], bn_mult[idx:idx+3]
    )
    idx += 4; mom_idx += 8; 

    for i in 2:size(rn.layer4.layers, 1)
        rn.layer4.layers[i] = load_mat_weights(
            rn.layer4.layers[i], conv_w[idx:idx+2], bn_mom[mom_idx:mom_idx+5], bn_b[idx:idx+2], bn_mult[idx:idx+2]
        )
        idx += 3; mom_idx += 6; 
    end
    
    # setting layer 5
    rn.layer5.layers[1] = load_mat_weights(
        rn.layer5.layers[1], conv_w[idx:idx+3], bn_mom[mom_idx:mom_idx+7], bn_b[idx:idx+3], bn_mult[idx:idx+3]
    )
    idx += 4; mom_idx += 8;

    for i in 2:size(rn.layer5.layers, 1)
        rn.layer5.layers[i] = load_mat_weights(
            rn.layer5.layers[i], conv_w[idx:idx+2], bn_mom[mom_idx:mom_idx+5], bn_b[idx:idx+2], bn_mult[idx:idx+2]
        )
        idx += 3; mom_idx += 6; 
    end  
    print("The pretrained ResNet50 model is successfully loaded!\n")
    return rn
end

function _get_params(path)
    wr50 = matread(path)
    params = wr50["params"]
    total_params = size(params["value"], 2)
    
    classes_imagenet = wr50["meta"]["classes"]["description"]
    avg_img = wr50["meta"]["normalization"]["averageImage"]
    
    conv_w = []; conv_b = [];
    fc_w = nothing; fc_b = nothing;
    bn_mom = []; bn_b = []; bn_mult = [];
    
    for k in 1:total_params
        name = params["name"][k]
        value = convert(Array{Float32}, params["value"][k])
        if endswith(name, "moments")
            push!(bn_mom, reshape(value[:,1], (1,1,size(value,1),1)))
            push!(bn_mom, reshape(value[:,2], (1,1,size(value,1),1)))
        elseif startswith(name, "bn") && endswith(name, "bias")
            push!(bn_b, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "bn") 
            push!(bn_mult, reshape(value, (1,1,length(value),1)))
        elseif startswith(name, "fc") && endswith(name, "filter")
            fc_w = reshape(value,(size(value,3),size(value,4)))'
        elseif startswith(name, "fc") && endswith(name, "bias")
            fc_b = value
        elseif endswith(name, "filter")
            push!(conv_w, value)
        elseif endswith(name, "bias")
            conv_b = reshape(value, (1,1,length(value),1))
        end
    end
    return conv_w, conv_b, fc_w, fc_b, bn_mom, bn_b, bn_mult, classes_imagenet
end 


















