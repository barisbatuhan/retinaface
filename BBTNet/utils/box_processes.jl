include("../../configs.jl")

"""
The parts below are mostly adapted from:
* https://github.com/biubug6/Pytorch_Retinaface
* https://github.com/Hakuyume/chainer-ssd
"""



""" 
Matches 2 boxes with each other. 
As context_head input, place 1 for first head and 2 for second head
"""
function match_boxes(boxes1, boxes2, context_head; dtype=Array{Float64})
    N, B, _ = size(bboxes2)
    class_fill = convert(dtype, zeros(N, B, 2))
    bbox_fill = convert(dtype, zeros(N, B, 4))
    landmark_fill = convert(dtype, zeros(N, B, 10))
    pos_thr = head2_pos_iou; neg_thr = head2_neg_iou
    if context_head == 1 
        pos_thr = head1_pos_iou; neg_thr = head1_neg_iou
    end
    # got intersection of union for each box combination
    for n in 1:N
        iou_vals = iou(boxes1[n,:,:], _to_min_max_form(boxes2[n,:,:]))
        max_prior_vals, max_prior_idx = findmax(iou_vals; dims=2)
        max_gt_vals, max_gt_idx = findmax(iou_vals; dims=1)
        

    end
    return class_fill, bbox_fill, landmark_fill
end


"""
!!! Each individual box in boxes has the format (x_min, y_min, x_max, y_max) !!!
Returns intersection of unions for combination of each boxes in both parameters.
A   : number of boxes in boxes1
B   : number of boxes in boxes2

Return: iou values with shape (A, B)
"""
function iou(boxes1, boxes2)
    A, C = size(boxes1); B, D = size(boxes2)
    intersections = _get_intersections(boxes1, boxes2)
    
    area1 = (boxes1[:,4] .- boxes1[:,2]) .* (boxes1[:,3] .- boxes1[:,1])
    area2 = (boxes2[:,4] .- boxes2[:,2]) .* (boxes2[:,3] .- boxes2[:,1])

    area1 = repeat(reshape(area1, (A, 1)), outer=(1, B))
    area2 = repeat(reshape(area1, (1, B)), outer=(A, 1))
    unions = area1 .+ area2 .- intersections

    return intersections ./ unions
end

"""
Conversion from: 
(center_x, center_y, width, height) --> (min_x, min_y, max_x, max_y)
If the start of a box is negative or bigger than the image size, then 
it is set to either 0 or image size.
"""
function _to_min_max_form(boxes)
    half_lens = boxes[:,3:4] ./ 2
    return cat(
        max.(boxes[:,1:2] .- half_lens, 0), 
        min.(boxes[:,1:2] .+ half_lens, img_size), dims=2)
end

"""
Conversion from: 
(min_x, min_y, max_x, max_y) --> (center_x, center_y, width, height)
"""
function _to_center_length_form(boxes)
    lengths = boxes[:,3:4] .- boxes[:,1:2]
    return cat(boxes[:,1:2] .+ (lengths ./ 2), lengths, dims=2)
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
    A, C = size(boxes1)
    B, D = size(boxes2)
    b1_mins = repeat(reshape(boxes1[:, 1:2], (A, 1, 2)), outer=(1, B, 1))
    b2_mins = repeat(reshape(boxes2[:, 1:2], (1, B, 2)), outer=(A, 1, 1))
    min_coords = min.(b1_mins, b2_mins)
    
    b1_maxs = repeat(reshape(boxes1[:, 3:4], (A, 1, 2)), outer=(1, B, 1))
    b2_maxs = repeat(reshape(boxes2[:, 3:4], (1, B, 2)), outer=(A, 1, 1))
    max_coords = max.(b1_maxs, b2_maxs)
    
    intersections = max.(max_coords .- min_coords, 0)
    return intersections[:,:,1] .* intersections[:,:,2]
end

"""
Returns the anchor boxes with their center_x, center_y, width, height information.
"""
function _get_priorboxes(;dtype=Array{Float64})
    feature_maps = [Int(ceil(img_size / scale["stride"])) for scale in anchor_info]
    num_proposals = num_anchors * sum([i*i for i in feature_maps])
    anchors = convert(dtype, zeros(num_proposals, 4))

    counter = 1
    for (idx, f) in enumerate(feature_maps)
        scaler = anchor_info[idx]["stride"] / img_size
        for s in anchor_info[idx]["anchors"]
            bbox_len = s / img_size
            for h in 1:f
                for w in 1:f
                    cx = (w - 0.5) * scaler
                    cy = (h - 0.5) * scaler
                    anchors[counter,:] = convert(dtype, [cx, cy, bbox_len, bbox_len])
                    counter += 1
                end
            end
        end
    end
    return anchors
end

"""
Decoder functions:
* The main motivation is that the network does not directly predict the proposals but
the combination of prediction and prior anchor box style and rescaling gives the actual
bounding box and landmark coordinations.
"""
function decode_points(bboxes, landmarks; dtype=Array{Float64})
    priors = _get_priorboxes(dtype=dtype)
    decoded_bboxes = _decode_bboxes(bboxes, priors)
    decoded_bboxes .*= img_size
    decoded_landmarks = _decode_landmarks(landmarks, priors)
    decoded_landmarks .*= img_size
    return decoded_bboxes, decoded_landmarks
end

function _decode_bboxes(bbox, priors)
    P = size(priors)[1]
    print(typeof(bbox), typeof(priors), typeof(variances[1]))
    centers = reshape(priors[:, 1:2], (1, P, 2)) .+ (bbox[:, :, 1:2] .* variances[1]) .* reshape(priors[:, 3:end], (1, P, 2))
    lengths = reshape(priors[:, 3:end], (1, P, 2)) .* exp.(bbox[:, :, 3:end] .* variances[2])
    centers .-= centers ./ 2
    return cat(centers, lengths, dims=3)
end

function _decode_landmarks(landmarks, priors)
    P = size(priors)[1]
    lm1 = reshape(priors[:, 1:2], (1, P, 2)) .+ landmarks[:, :, 1:2] .* variances[1] .* reshape(priors[:, 3:end], (1, P, 2))
    lm2 = reshape(priors[:, 1:2], (1, P, 2)) .+ landmarks[:, :, 3:4] .* variances[1] .* reshape(priors[:, 3:end], (1, P, 2))
    lm3 = reshape(priors[:, 1:2], (1, P, 2)) .+ landmarks[:, :, 5:6] .* variances[1] .* reshape(priors[:, 3:end], (1, P, 2))
    lm4 = reshape(priors[:, 1:2], (1, P, 2)) .+ landmarks[:, :, 7:8] .* variances[1] .* reshape(priors[:, 3:end], (1, P, 2))
    lm5 = reshape(priors[:, 1:2], (1, P, 2)) .+ landmarks[:, :, 9:10] .* variances[1] .* reshape(priors[:, 3:end], (1, P, 2))
    return cat(lm1, lm2, lm3, lm4, lm5, dims=3)
end