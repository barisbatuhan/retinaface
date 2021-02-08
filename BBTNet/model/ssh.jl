include("../../configs.jl")

"""
Non-Deformable Context Head Module Network components for the RetinaFace. 
"""
mutable struct SSH_Detect_Module 
    conv128; 
    conv64_1_1; conv64_1_2; 
    conv64_2_1; conv64_2_2; 
end

function SSH_Detect_Module(; input_dim=256)
    out_dim_4 = Int(floor(input_dim/4))
    out_dim_2 = Int(floor(input_dim/2))
    return SSH_Detect_Module(
        ConvBnRelu(input_dim, out_dim_2, 3, bias=false, padding=1, leaky=nothing),
        
        ConvBnRelu(input_dim, out_dim_4, 3, bias=false, padding=1),
        ConvBnRelu(out_dim_4, out_dim_4, 3, bias=false, padding=1, leaky=nothing),

        ConvBnRelu(out_dim_4, out_dim_4, 3, bias=false, padding=1),
        ConvBnRelu(out_dim_4, out_dim_4, 3, bias=false, padding=1, leaky=nothing),
    )
end

function (ssh::SSH_Detect_Module)(x)
    o1 = ssh.conv128(x)
    o2_1 = ssh.conv64_1_1(x)
    o2_2 = ssh.conv64_1_2(o2_1)
    o3_1 = ssh.conv64_2_1(o2_1)
    o3_2 = ssh.conv64_2_2(o3_1)
    combined = cat(o1, o2_2, o3_2, dims=3)
    return relu.(combined)
end