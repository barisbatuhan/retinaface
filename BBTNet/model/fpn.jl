include("../../configs.jl")

"""
Upsampling and Pyramid Network for the RetinaFace. 
"""
mutable struct FPN lateral_ins; lateral_outs; scale_cnt; extra::Bool; end

function FPN(;in_sizes=[256, 512, 1024, 2048], out_size=256, scale_cnt=5)
    
    lateral_ins = []
    for ins in in_sizes
        push!(lateral_ins, ConvBnRelu(ins, out_size, 1, bias=false)) 
    end
    extra = false
    if length(in_sizes) < scale_cnt
        extra = true
        push!(lateral_ins, ConvBnRelu(
            in_sizes[end], out_size, 3, stride=2, padding=1, bias=false, init=xavier))
    end

    lateral_outs = []
    for _ in 1:length(lateral_ins)-1
        push!(lateral_outs, ConvBnRelu(
            out_size, out_size, 3, bias=false, padding=1))
    end

    return FPN(lateral_ins, lateral_outs, scale_cnt, extra)
end

function (fpn::FPN)(xs)
    outs = []; lateral_size = fpn.scale_cnt;
    # extra conv layer
    if fpn.extra
        push!(outs, fpn.lateral_ins[end](xs[end]))
        lateral_size -= 1;
    end

    inter_outs = []
    for idx in 1:lateral_size
        push!(inter_outs, fpn.lateral_ins[idx](xs[idx]))
    end

    push!(outs, inter_outs[end])
    for idx in lateral_size:-1:2
        conn = unpool(inter_outs[idx]) + inter_outs[idx-1]
        push!(outs, fpn.lateral_outs[idx-1](conn))
    end

    return reverse(outs) # to protect the input order
end