using Knet

function smooth_l1(y, pred; beta=1)
    diff = abs.(y .- pred)
    cpu_diff = Array(value(diff))
    low_idx = findall(cpu_diff .< beta)
    high_idx = findall(cpu_diff .>= beta)
    loss_sum = 0
    loss_sum += sum(0.5 .* (diff[low_idx].^2) ./ beta)
    loss_sum += sum(diff[high_idx] .- (0.5 * beta))
    return sum(loss_sum) / size(diff, 1)
end