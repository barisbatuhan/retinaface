using Knet

function smooth_l1(y, pred; beta=1)
    diff = abs.(y .- pred)
    diff[findall(diff .< beta)] = 0.5 * (diff[findall(diff .< 1)].^2) / beta
    diff[findall(diff .>= beta)] = diff[findall(diff .>= 1)] .- (0.5 * beta)
    return sum(diff)
end