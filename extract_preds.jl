using ArgParse, MAT

include("BBTNet/model/retinaface.jl")
include("BBTNet/datasets/WIDERFACE.jl")
include("configs.jl")
include("./DeepJulia/DeepJulia.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--load_dir"
            help = "If there is a pretrained model, then the path of it."
            arg_type = String
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
        "--save_dir"
            help = "Directory to save all prediction files."
            arg_type = String
            default = "./data/wider_txt/"
        "--nms"
            help = "NMS threshold, for eliminating redundant boxes for computational effiency."
            arg_type = Float64
            default = 0.4
        "--conf"
            help = "Confidence threshold, for eliminating really low probability boxes for computational effiency."
            arg_type = Float64
            default = 0.02
    end  
    return parse_args(s) 
end

function main()
    parsed_args = parse_cmd()
    nms_threshold = parsed_args["nms"]; conf_threshold = parsed_args["conf"]; 
    mode = parsed_args["mode"]; scale_cnt = parsed_args["laterals"];
    num_anchors = scale_cnt == 3 ? 2 : 3
    anchor_info = scale_cnt == 3 ? lat_3_anchors : lat_5_anchors
    
    load_path = parsed_args["load_dir"]; eval_dir = parsed_args["eval_dir"];
    save_dir = parsed_args["save_dir"]; 
    rm(save_dir, force=true, recursive=true); mkdir(save_dir);
    model = RetinaFace(
        mode=mode, num_anchors=num_anchors, anchor_info=anchor_info #, load_path=load_path
    )
    model = set_eval_mode(model)
    if run_gpu 
        model = to_gpu(model)
    end
    print("[INFO] Model is loaded!\n")
    
    val_data = matread(eval_dir * "wider_face_val.mat") 
    events = val_data["event_list"]; esize = size(events, 1); event_files = val_data["file_list"];

    img_paths = []; files = [];
    for e in 1:esize
        event = events[e, 1]
        for f in 1:length(event_files[e])
            file = event_files[e][f]
            img_path = wf_path  * "val/images/" * event * "/" * file * ".jpg" 
            push!(files, [event, file])
            push!(img_paths, img_path)
        end
    end

    d = WIDER_Data(img_paths, nothing, train=false, shuffle=false, batch_size=1); 
    d.tr.img_paths = img_paths;
    imgs, labels = iterate(d, restart=true); 
    if run_gpu; imgs = convert(atype, imgs); end;
    iterate_cnt = 1;
    
    while imgs !== nothing
        img_path = d.tr.img_paths[iterate_cnt]
        img_h, img_w = size(load(img_path)); 
        
        resized_len = size(imgs, 3); scaler = max(img_w, img_h) / resized_len;
        c, b, l = predict_image(model, imgs, mode=mode, verbose=false, nms_thold=0.2, conf_thold=conf_threshold)
        b .*= scaler; l .*= scaler; # resize to original size
        # excluding zero fills to convert image to square
        if img_w < img_h
            b[1:2:3,:] .-= floor((img_h - img_w) / 2)
        elseif img_h < img_w
            b[2:2:4,:] .-= floor((img_w - img_h) / 2)
        end

        invalids = 0
        for person in 1:size(b, 2)
            x1, y1, x2, y2 = round.(Int, b[:,person])
            if x2 <= 0 || y2 <= 0 || x1 >= img_w || y1 >= img_h
                invalids += 1
            end
        end
            
        logfile = save_dir * files[iterate_cnt][1] * "/" * files[iterate_cnt][2] * ".txt"
        mkpath(save_dir * files[iterate_cnt][1]); 
        open(logfile, "w") do io 
		write(io, files[iterate_cnt][2] * "\n" * string(size(b, 2) - invalids) * "\n") 
            for person in 1:size(b, 2)
                x1, y1, x2, y2 = round.(Int, b[:,person]); w = x2 - x1; h = y2 - y1;
                if x2 <= 0 || y2 <= 0 || x1 >= img_w || y1 >= img_h
                    continue
                end
                conf = c[2,person]
                to_print = string(x1) * " " * string(y1) * " " * string(w) * " " * string(h) * " " * string(conf) * "\n"
                write(io, to_print) 
            end
        end
        println("Iteration: ", iterate_cnt, " finished!");
        imgs, _ = iterate(d); iterate_cnt += 1;
        if run_gpu; imgs = convert(atype, imgs); end;
    end
end

main()
