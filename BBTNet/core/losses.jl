using Knet

function smooth_l1(y, pred; beta=1)
    diff = abs.(y .- pred)
    cpu_diff = Array(value(diff))
    low_idx = findall(cpu_diff .< beta)
    high_idx = findall(cpu_diff .>= beta)
    diff[low_idx] = 0.5 .* (diff[low_idx].^2) ./ beta
    diff[high_idx] = diff[high_idx] .- (0.5 * beta)
    return sum(diff)
end