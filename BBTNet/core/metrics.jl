using MAT

include("../../configs.jl")

"""
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

"""

function AP(gt, bboxes, confidences; iou_thold=0.5, thresh_num=1000, iou_vals=nothing, return_array=false)
    results = zeros(2, thresh_num+1) # precisions 1st idx, recalls in 2nd idx
    if iou_vals === nothing iou_vals = iou(gt[1:4,:], bboxes) end
    
    selected_boxes = zeros(size(bboxes, 2)); selected_gts = zeros(size(gt, 2)); assigned_gts = 0;
    
    pos_ious = findall(iou_vals .>= iou_thold)
    pos_priors = getindex.(pos_ious, [1 2])[:,1]; pos_gts = getindex.(pos_ious, [1 2])[:,2];
    sorted_priors = sortperm(confidences[2,pos_priors], rev=true)
   
    neg_ious = findall(iou_vals .< iou_thold); assigned_neg_gts = 0; 
    neg_priors = getindex.(neg_ious, [1 2])[:,1]; sorted_neg_priors = sortperm(confidences[2,neg_priors], rev=true);
    neg_gts = getindex.(neg_ious, [1 2])[:,2]; num_neg_gts = length(unique(neg_gts));
    
    for t in 1:thresh_num+1
        thold = (t-1) / thresh_num
        TP = 0; FP = 0; FN = 0;
        for s in sorted_priors
            prior = pos_priors[s]; gt_idx = pos_gts[s];  
            if confidences[2, prior] >= thold && selected_boxes[prior] == 0 && selected_gts[gt_idx] == 0
                # found TP example
                TP += 1; assigned_gts += 1;  
                selected_boxes[prior] = 1; selected_gts[gt_idx] = 1;
            elseif confidences[2, prior] < thold
                # unassigned positive gt boxes are counted as FN
                FN = length(unique(pos_gts)) - assigned_gts 
                break
            end
        end
        
        for ns in sorted_neg_priors
            prior = neg_priors[ns]; gt_idx = neg_gts[ns];  
            if confidences[2, prior] >= thold && selected_boxes[prior] == 0 && selected_gts[gt_idx] == 0
                # found TP example
                FP += 1; assigned_neg_gts += 1;  
                selected_boxes[prior] = -1; selected_gts[gt_idx] = -1;
            elseif confidences[2, prior] < thold || num_neg_gts == assigned_neg_gts
                break
            end
        end
        
        precision = TP == 0 ? 0 : TP / (TP + FP)
        recall = TP == 0 ? 0 : TP / (TP + FN)
        results[1, t] = precision; results[2, t] = recall;
    end

    main_results = zeros(thresh_num+1)
    for t in 1:thresh_num+1
        thold = (t-1) / thresh_num
        recall_indices = findall(results[2,:] .>= thold)
        if length(recall_indices) == 0
            main_results[t] = 0
        else
            main_results[t] = maximum(results[1,recall_indices])
        end
    end
    
    if return_array return main_results end
    
    AP = sum(main_results) / (thresh_num + 1)
    return AP
end

function mAP(gt, bboxes, confidences, iou_tholds; thresh_num=1000)
    APs = 0; cnt = 0; iou_vals = iou(gt, bboxes);
    for iou_thold in iou_tholds[1]:iou_tholds[2]:iou_tholds[3]
        APs += AP(gt, bboxes, confidences, iou_vals=iou_vals, thresh_num=1000, iou_thold=iou_thold)
        cnt += 1
    end
    mAP = APs / cnt
    return mAP
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