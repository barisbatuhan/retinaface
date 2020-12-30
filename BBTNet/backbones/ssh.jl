include("../core/layers.jl")
include("../core/blocks.jl")
include("../../configs.jl")

"""
Non-Deformable Context Head Module Network components for the RetinaFace. 
"""
struct SSH conv128; conv64_1_1; conv64_1_2; conv64_2_1; conv64_2_2; end

function SSH(; input_dim=256, dtype=Array{Float32})
    out_dim = Int(floor(input_dim/4))
    return SSH(
        ConvBn(3, 3, input_dim, Int(floor(input_dim/2)), bias=false, dtype=dtype, padding=1),
        
        ConvBn(3, 3, input_dim, out_dim, padding=1, bias=false, dtype=dtype, f=relu),
        ConvBn(3, 3, out_dim, out_dim, padding=1, bias=false, dtype=dtype),
        
        ConvBn(3, 3, out_dim, out_dim, padding=1, bias=false, dtype=dtype, f=relu),
        ConvBn(3, 3, out_dim, out_dim, padding=1, bias=false, dtype=dtype)
    )
end

function (ssh::SSH)(x; train=true)
    o1 = ssh.conv128(x, train=train)
    o2_1 = ssh.conv64_1_1(x, train=train)
    o2_2 = ssh.conv64_1_2(o2_1, train=train)
    o3_1 = ssh.conv64_2_1(o2_1, train=train)
    o3_2 = ssh.conv64_2_2(o3_1, train=train)
    combined = cat(o1, o2_2, o3_2, dims=3)
    return relu.(combined)
end