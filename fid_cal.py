from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import torch
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--new_data_dir", type=str, default="samples")
    parse.add_argument("--ref_data_dir", type=str, default="../all_imgs/cifar10")
    args = parse.parse_args()
    BATCH_SIZE=128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.new_data_dir):
        os.makedirs(args.new_data_dir)
        
    paths = [args.new_data_dir, args.ref_data_dir]
    print("#generated images: {:d}, #reference images: {:d}".format(
        len(os.listdir(args.new_data_dir)), len(os.listdir(args.ref_data_dir))))

    try:
        fid_score = calculate_fid_given_paths(paths, BATCH_SIZE, device, dims=192)
        print("Dimension {:d} works! fid score:".format(192, fid_score, args.new_data_dir))
    except:
        print("Dimension {:d} fails!".format(192))
        
    print(f"{args.ref_data_dir} and {args.new_data_dir} " + "Average fid score: {}".format(fid_score))