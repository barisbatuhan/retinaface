include("../../configs.jl")

"""
!!! Each individual box in boxes has the format (x_min, y_min, x_max, y_max) !!!
Returns intersection of unions for combination of each boxes in both parameters.
A   : number of boxes in boxes1
B   : number of boxes in boxes2

Return: iou values with shape (A, B)
"""
function iou(boxes1, boxes2)
    C, A = size(boxes1); D, B = size(boxes2);
    area1 = max.(boxes1[4,:] .- boxes1[2,:], 0) .* max.(boxes1[3,:] .- boxes1[1,:], 0)
    area2 = max.(boxes2[4,:] .- boxes2[2,:], 0) .* max.(boxes2[3,:] .- boxes2[1,:], 0) 
    intersections = _get_intersections(boxes1, boxes2) 
    unions = reshape(area1, (1, A)) .+ reshape(area2, (B, 1)) .- intersections
    return intersections ./ unions
end

"""
!!! Each individual box in boxes has the format (x_min, y_min, x_max, y_max) !!!
We resize both boxes to [A,B,4]:
* [A,2] -> [A,1,2] -> [A,B,2]
* [B,2] -> [1,B,2] -> [A,B,2]
Then we compute the area of intersect between box_a and box_b.

Return: intersection area with shape (A, B)
"""
function _get_intersections(boxes1, boxes2)
    C, A = size(boxes1); D, B = size(boxes2);
    min_coords = max.(reshape(boxes1[1:2,:], (2, 1, A)), reshape(boxes2[1:2,:], (2, B, 1)))
    max_coords = min.(reshape(boxes1[3:4,:], (2, 1, A)), reshape(boxes2[3:4,:], (2, B, 1)))  
    intersections = max.(max_coords .- min_coords, 0)
    return intersections[1,:,:] .* intersections[2,:,:]
end