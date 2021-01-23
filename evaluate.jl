using ArgParse, MAT

include("BBTNet/models/retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("./BBTNet/utils/ImageReader.jl")
include("./BBTNet/core/metrics.jl")
include("configs.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--load_dir"
            help = "If there is a pretrained model, then the path of it."
            arg_type = String
            required = true
        "--log_file", "-s"
            help = "Log path for saving a the result after evaluation."
            arg_type = String
            default = "./logs/evaluations.txt"
            required = false
        "--mode", "-m"
            help = "Training mode: 0 for only baseline until FPN, 1 for full model, 2 for no cascaded structure."
            arg_type = Int
            default = 1
            required = false
        "--laterals", "-l"
            help = "How many lateral connections will be processed, either 3 or 5. 5 is needed for full model."
            arg_type = Int
            default = 5
            required = false
        "--eval_dir"
            help = "WIDER FACE Easy/Medium/Hard .mat files directory path.."
            arg_type = String
            required = true
    end  
    return parse_args(s) 
end

function get_full_names(mfile)
    events = mfile["event_list"]; esize = size(events, 1);
    files = mfile["file_list"]; file_arr = [];
    for e in 1:esize
        event = events[e, 1]
        for file in files[e, 1]
            push!(file_arr, event * "/" * file * ".jpg")
        end
    end
    return file_arr
end

function main()
    parsed_args = parse_cmd()
    mode = parsed_args["mode"]
    scale_cnt = parsed_args["laterals"]
    num_anchors = scale_cnt == 3 ? 2 : 3
    anchor_info = scale_cnt == 3 ? lat_3_anchors : lat_5_anchors
    
    load_path = parsed_args["load_dir"]; log_file = parsed_args["log_file"]; eval_dir = parsed_args["eval_dir"];
    model = RetinaFace(
        mode=mode, num_anchors=num_anchors, anchor_info=anchor_info, load_path=load_path, dtype=atype
    )
    print("[INFO] Model is loaded!\n")
    
    val_dir = wf_path * "val/"
    labels_dir = wf_labels_path * "val/"
    d = WIDER_Data(val_dir, labels_dir, train=false, shuffle=false, batch_size=1, dtype=atype)
    print("[INFO] Data is loaded!\n")
    
    hard = matread(eval_dir * "wider_hard_val.mat")
    med = matread(eval_dir * "wider_medium_val.mat")
    easy = matread(eval_dir * "wider_easy_val.mat")
    
    events = hard["event_list"]; esize = size(events, 1); files = hard["file_list"];
    APs = zeros(3); mAP_hard = 0; cnts = zeros(3); total_cnt = 0;
    for e in 1:esize
        event = events[e, 1]
        for f in 1:length(files[e])
            file = files[e][f]
            path = event * "/" * file * ".jpg"; img_path = val_dir * "images/" * path;
            # box numbers are retrieved
            hard_gt = Int64.(hard["gt_list"][e][f])
            med_gt = Int64.(med["gt_list"][e][f])
            easy_gt = Int64.(easy["gt_list"][e][f])
            # reading the image
            img, boxes = read_img(img_path, boxes=d.bboxes[path]) 
            reversed_img = reshape(reverse(img, dims=1), (size(img)..., 1))
            imgs_arr = convert(atype, permutedims(reversed_img, (3,2,1,4)))
            # predicting the values
            c, b, l = predict_image(model, imgs_arr, mode=mode, verbose=false)
            # evaluate easy
            if length(easy_gt) > 0
                AP_val = 0
                if typeof(easy_gt) <: Array
                    AP_val = AP(boxes[:,easy_gt], b, c; iou_thold=0.5, thresh_num=1000)
                else
                    AP_val = AP(boxes[:,easy_gt:easy_gt], b, c; iou_thold=0.5, thresh_num=1000)
                end
                APs[1] += AP_val; cnts[1] += 1; 
            end
            # evaluate medium
            if length(med_gt) > 0
                AP_val = 0
                if typeof(med_gt) <: Array
                    AP_val = AP(boxes[:,med_gt], b, c; iou_thold=0.5, thresh_num=1000)
                else
                    AP_val = AP(boxes[:,med_gt:med_gt], b, c; iou_thold=0.5, thresh_num=1000)
                end
                APs[2] += AP_val; cnts[2] += 1; 
            end
            # evaluate hard
            if length(hard_gt) > 0
                AP_val = 0.0; mAP_val  = 0.0;
                if typeof(hard_gt) <: Array
                    AP_val += AP(boxes[:,hard_gt], b, c; iou_thold=0.5, thresh_num=1000)
                    mAP_val += mAP(boxes[:,hard_gt], b, c, mAP_ious, thresh_num=1000)
                else
                    AP_val += AP(boxes[:,hard_gt:hard_gt], b, c; iou_thold=0.5, thresh_num=1000)
                    mAP_val += mAP(boxes[:,hard_gt:hard_gt], b, c, mAP_ious, thresh_num=1000)
                end
                APs[3] += AP_val
                mAP_hard += mAP_val
                cnts[3] += 1
            end
            
            total_cnt += 1
            
            AP_easy = round(APs[1] / max(1, cnts[1]), digits=4); 
            AP_med = round(APs[2] / max(1, cnts[2]), digits=4);
            AP_hard = round(APs[3] / max(1, cnts[3]), digits=4);
            mAP_h = round(mAP_hard / max(1, cnts[3]), digits=4);
            to_print = "Img: " * string(total_cnt) * " / " * string(length(d.files)) 
            to_print *= " -->  APs = Easy: " * string(AP_easy) * " | Med: " * string(AP_med)
            to_print *= " | Hard: " *string(AP_hard) * " && mAP Hard: " * string(mAP_h) * "\n"
            print(to_print)
            if log_file != nothing
                open(log_file, "a") do io write(io, to_print) end;
            end
        end
    end 
end

main()
