using MAT

include("../../configs.jl")

"""
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

"""

function evaluate_model(model; dataset="widerface", metric="mAP", iou_tholds=[0.5,0.05,0.95])
    val_mat = matread("/datasets/widerface/wider_face_split/wider_face_val.mat")
    facebox_list = val_mat["face_bbx_list"]
    event_list = val_mat["event_list"]
    file_list = val_mat["file_list"]
    
    
end



function meanAP(gt, bboxes, confidences, iou_tholds; conf_thold=0.5)
    conf_delta = 1000; num_iters = 0; num_faces = size(gt, 2); num_boxes = size(bboxes, 2);
    curve = Dict() # keys are iou vals, values are [precision, recall]
    
    iou_vals = iou(gt, bboxes)
    # gets max values and indices for each gt
    max_gt_vals, max_gt_idx = findmax(iou_vals; dims=1)
    # gets max values and indices for each prior
    max_prior_vals, max_prior_idx = findmax(iou_vals; dims=2) 
    
    for thold in iou_tholds[1]:iou_tholds[2]:iou_tholds[3]
        TP = length(findall(max_gt_vals .>= thold))
        FN = num_faces - TP
        FP = 0
        for t in 1:conf_delta
            conf = 1 - t/conf_delta
            
        end
        
        num_iters += 1
    end
end

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