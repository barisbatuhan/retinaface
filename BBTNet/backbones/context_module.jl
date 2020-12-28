include("ssh.jl")
include("../../configs.jl")

"""
Non-Deformable Context Head Module Network for the RetinaFace. 
"""
mutable struct ContextModule ssh_p2; ssh_p3; ssh_p4; ssh_p5; ssh_p6; end

function ContextModule(; input_dim=256, dtype=Array{Float32})
    return ContextModule(
        SSH(input_dim=input_dim, dtype=dtype),
        SSH(input_dim=input_dim, dtype=dtype),
        SSH(input_dim=input_dim, dtype=dtype),
        SSH(input_dim=input_dim, dtype=dtype),
        SSH(input_dim=input_dim, dtype=dtype)
    )
end

function (ch::ContextModule)(xs; train=true)
    
    if scale_cnt == 5
        p2, p3, p4, p5, p6 = xs
        s2 = ch.ssh_p2(p2, train=train)
        s3 = ch.ssh_p3(p3, train=train)
        s4 = ch.ssh_p4(p4, train=train)
        s5 = ch.ssh_p5(p5, train=train)
        s6 = ch.ssh_p6(p6, train=train)
        return [s2, s3, s4, s5, s6]
    
    elseif scale_cnt == 3
        p3, p4, p5 = xs
        s3 = ch.ssh_p3(p3, train=train)
        s4 = ch.ssh_p4(p4, train=train)
        s5 = ch.ssh_p5(p5, train=train)
        return [s3, s4, s5]
    end
    
    return nothing
end