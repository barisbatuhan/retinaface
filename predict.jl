using ArgParse

include("BBTNet/models/retinaface.jl")
include("./BBTNet/utils/ImageReader.jl")
include("./BBTNet/utils/draw.jl")
include("configs.jl")

function parse_cmd()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--image_path"
            help = "Path of the image to predict."
            arg_type = String
            required = true
        "--load_dir"
            help = "If there is a pretrained model, then the path of it."
            arg_type = String
            required = true
        "--save_dir", "-s"
            help = "Path for saving a the result after prediction."
            arg_type = String
            default = "./data/results/predicted.png"
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
    end  
    return parse_args(s) 
end

function main()
    parsed_args = parse_cmd()
    img_path = parsed_args["image_path"]
    mode = parsed_args["mode"]
    scale_cnt = parsed_args["laterals"]
    num_anchors = scale_cnt == 3 ? 2 : 3
    anchor_info = scale_cnt == 3 ? lat_3_anchors : lat_5_anchors
    
    load_path = parsed_args["load_dir"]; save_path = parsed_args["save_dir"];
    model = RetinaFace(
        mode=mode, num_anchors=num_anchors, anchor_info=anchor_info, load_path=load_path, dtype=atype
    )
    print("[INFO] Model is loaded!\n")
    
    img, _ = read_img(img_path); reversed_img = reshape(reverse(img, dims=1), (size(img)..., 1));
    imgs_arr = convert(atype, permutedims(reversed_img, (3,2,1,4)))
    c, b, l = predict_image(model, imgs_arr, mode=mode); b[b .< 0] .= 0; l[l .< 0] .= 0;
    
    init_img = (img .+ avg_img) ./ 255
    # return draw_boxes_and_landmarks(init_img, b, l; len=640)
    
    if save_path !== nothing
        init_img = (img .+ avg_img) ./ 255
        plotted = draw_boxes_and_landmarks(init_img, b, l; conf=c, len=640)
        png(plotted, save_path)
    end
    
    println("# of persons found:", size(c, 1))
    
    return c, b, l
end

main()