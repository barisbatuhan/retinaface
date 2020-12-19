using Knet

# if mode is 1, then fanin will be used, otherwise fanout
# Mostly taken from Knet source code
function kaiming_uniform(a...; mode=1, gain=1)
    w = rand(a...)
    if ndims(w) == 1
        fanout = 1
        fanin = length(w)
    elseif ndims(w) == 2
        fanout = size(w,1)
        fanin = size(w,2)
    else
        fanin = div(length(w),  a[end])
        fanout = div(length(w), a[end-1])
    end
    
    fan = fanin
    if mode != 1 fan = fanout end
    
    s = convert(eltype(w), gain*sqrt(3 / (fan)))
    return 2s .* w .- s
end