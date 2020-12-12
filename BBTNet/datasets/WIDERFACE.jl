using Random
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take

include("../utils/ImageReader.jl")
include("../../configs.jl")

mutable struct WIDER_Data
    dir::String
    bboxes
    files::Array{String}
    batch_size::Int64
    num_files::Int64
    num_faces::Int64
    shuffle::Bool
    augment::Bool
    curr_idx::Int64
    dtype
    
    function WIDER_Data(dir; batch_size::Int64=32, train::Bool=true, shuffle::Bool=true, dtype=Array{Float32})
        files = []
        bbox_dict = Dict()
        num_faces = 0
        open(dir * "label.txt", "r") do io
            lines = readlines(io)
            filename = nothing
            for line in lines
                if line[1] == '#'
                    if filename !== nothing
                        bbox_dict[filename] = _clean_bboxes(bbox_dict[filename])
                    end
                    filename = line[3:end]
                    push!(files, filename)
                    bbox_dict[filename] = []
                else
                    bbox = [parse(Float64, x) for x in split(line, " ")]
                    num_faces += 1
                    push!(bbox_dict[filename], bbox)
                end
            end
            bbox_dict[filename] = _clean_bboxes(bbox_dict[filename])
        end
        
        return new(dir, bbox_dict, files, batch_size, length(files), num_faces, shuffle, train, 1, dtype)
    end
end

function _clean_bboxes(bboxes)
    annotations = zeros(15, length(bboxes))
    # annotation processing
    for person in 1:length(bboxes)
        z_coords = 0
        for iter in 1:length(bboxes[person])-1
            if (iter > 4) && (mod(iter, 3) == 1)
                z_coords += 1
                continue
            end
            i = iter - z_coords
            annotations[i,person] =  bboxes[person][iter]
        end
        if annotations[5,person] < 0 # no landmark info
            annotations[15,person] = -1.0
        else
            annotations[15,person] = 1.0
        end
    end
    return annotations
end

function iterate(data::WIDER_Data, state=ifelse(
    data.shuffle, randperm(data.num_files), collect(1:data.num_files)))
    
    r = Image_Reader(data.augment)
    if length(state) < data.batch_size || state === nothing
        return nothing
    else 
        imgs = data.files[state[1:data.batch_size]]
        imgs_arr = zeros(3, img_size, img_size, data.batch_size)
        labels = []
        idx = 1
        for img_path in imgs
            img_dir = data.dir * "images/" * img_path
            img, box = read_img(r, img_dir, data.bboxes[img_path], img_size)
            push!(labels, convert(data.dtype, box))
            imgs_arr[:,:,:,idx] .= img
            idx += 1
        end
        imgs_arr = convert(data.dtype, permutedims(imgs_arr, (3,2,1,4)))
        return (imgs_arr, labels), state[data.batch_size+1:end]
    end
end