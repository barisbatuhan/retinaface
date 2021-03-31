using Knet, CUDA

"""
ReLU object definition. If alpha is 0 then standard ReLU, otherwise Leaky ReLU
will be applied.
"""
struct ReLUs alpha; end

function ReLUs(;alpha=0)
    return ReLUs(alpha)
end

function (r::ReLUs)(x)
    if r.alpha == 0 relu(x)
    else max(r.alpha .* x, x) # Leaky ReLU
    end
end