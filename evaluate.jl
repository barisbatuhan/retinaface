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
    
#     hard = matread(eval_dir * "wider_hard_val.mat")
#     med = matread(eval_dir * "wider_medium_val.mat")
#     easy = matread(eval_dir * "wider_easy_val.mat")
#     files = get_full_names(hard)
    
    val_dir = wf_path * "val/"
    labels_dir = wf_labels_path * "val/"
    d = WIDER_Data(val_dir, labels_dir, train=false, shuffle=true, batch_size=1, dtype=atype)
    print("[INFO] Data is loaded!\n")
    
    APs = 0; mAPs = 0; cnt = 0;
    for file in d.files
        img_paths = val_dir * "images/" * file
        img, boxes = read_img(img_paths, boxes=d.bboxes[file]) 
        reversed_img = reshape(reverse(img, dims=1), (size(img)..., 1))
        imgs_arr = convert(atype, permutedims(reversed_img, (3,2,1,4)))
        c, b, l = predict_image(model, imgs_arr, mode=mode, verbose=false)
        AP_val = AP(boxes, b, c; iou_thold=0.5, thresh_num=1000)
        mAP_val = mAP(boxes, b, c, mAP_ious, thresh_num=1000)
        APs += AP_val; mAPs += mAP_val; cnt += 1;
        
        to_print = "Img: " * string(cnt) * " / " * string(length(d.files)) * " --> AP: " * string(AP_val) * " & mAP: " * string(mAP_val) * "\n"
        print(to_print)
        if log_file != nothing
            open(log_file, "a") do io write(io, to_print) end;
        end
    end  
    to_print = "=== Total --> AP: " * string(APs / cnt) * " & mAP: " * string(mAPs / cnt) * "\n"
    print(to_print)
    if log_file != nothing
        open(log_file, "a") do io write(io, to_print) end;
    end
end

main()
