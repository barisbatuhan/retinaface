using Knet

"""
Smooth L1 Loss. 
- As stated in the Fast R-CNN paper, it is calculated as:

    for |x| < beta --> 0.5 * x^2
    otherwise      --> |x| - 0.5
    
    where x = gt - pred
"""

function smooth_l1(x; beta=1)
    low_idx = findall(Array(x) .< beta)
    high_idx = findall(Array(x) .>= beta)
    loss_sum = sum(x[high_idx] .- 0.5)
    loss_sum += sum(0.5 .* (x[low_idx].^2))
    return loss_sum
end


