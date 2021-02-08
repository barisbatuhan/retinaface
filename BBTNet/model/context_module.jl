include("ssh.jl")
include("../../configs.jl")

"""
Non-Deformable Context Head Module Network for the RetinaFace. 
"""
mutable struct ContextModule sshs; scale_cnt; end

function ContextModule(; input_dim=256, scale_cnt=5)
    sshs = []
    for _ in 1:scale_cnt
        push!(sshs, SSH_Detect_Module(input_dim=input_dim))
    end
    return ContextModule(sshs, scale_cnt)
end

function (ch::ContextModule)(xs)
    outs = []
    for idx in 1:ch.scale_cnt
        push!(outs, ch.sshs[idx](xs[idx]))
    end
    return outs
end