import argparse
import time
import logging
import os
from bomoto.config import get_cfg
from bomoto.engine import Engine
from cluster.utils.config import parse_args
import numpy as np

SMPL_MODEL_FOLDER = "/is/cluster/sbhor/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/"
STAR_MODEL_FOLDER = "/is/cluster/sbhor/STAR/star_1_1/star/STAR_NEUTRAL.npz"

args = parse_args()

# parser = argparse.ArgumentParser()
# parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
# bomoto_args = parser.parse_args()

# Update paths based on command line arguments
path_smpl_params = args.input_dir
path_save_star_parms = args.output_dir

# Ensure the output directory exists
if not os.path.exists(path_save_star_parms):
    os.makedirs(path_save_star_parms)
    print(f'Created directory {path_save_star_parms}.')
    
# Configure logging to write to a file within the specified directory
log_file_path = os.path.join(path_save_star_parms, 'conversion_log.log')
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
print(f'Starting the conversion process at location {log_file_path}.')
logging.info(f"Starting the conversion process at location {log_file_path}.")


all_files =[]
for dirpath, dirnames, filenames in os.walk(path_smpl_params):
    for filename in filenames:
        if filename.endswith(".npz"):
            all_files.append(os.path.join(dirpath, filename))
if args.cluster_batch_size is None:
    cluster_batch_size = len(all_files)
else:
    cluster_batch_size = args.cluster_batch_size

selected_file_idxs = list(range(args.cluster_start_idx * cluster_batch_size,
                                    args.cluster_start_idx * cluster_batch_size + cluster_batch_size))
print(f"running {len(selected_file_idxs)} files in the range {selected_file_idxs[0]}-{selected_file_idxs[-1]}")


for file_idx in selected_file_idxs:
    file_path = all_files[file_idx]
    print(f'Loading the SMPL Parameters {file_path}')
    logging.info(f'Loading the SMPL Parameters from {file_path}')
    output_subdir = os.path.join(path_save_star_parms, file_path.split('/')[-3], file_path.split('/')[-2])
    
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
        print(f'Created directory {output_subdir}.')
        logging.info(f'Created directory {output_subdir}.')
    else:
        if os.path.exists(os.path.join(output_subdir, f'{file_path.split("/")[-1].split(".")[0]}_star.npz')):
            print(f'File {file_path.split("/")[-1]} already exists in {output_subdir}. Skipping.')
            logging.info(f'File {file_path.split("/")[-1]} already exists in {output_subdir}. Skipping.')
            continue
        
    print(f'Output directory: {output_subdir}')
    logging.info(f'Output directory: {output_subdir}')
    
    
    cfg = get_cfg(args.cfg)
    
    #get the file_path directory
    file_path_dir = os.path.dirname(file_path)
    cfg.input.data.npz_files_dir = file_path_dir
    cfg.output.save_dir = output_subdir
    smpl = np.load(file_path, allow_pickle=True)
    cfg.batch_size = smpl['poses'].shape[0]
    
    engine = Engine(cfg)

    start_time = time.time()

    engine.run()

    print("Total time:", time.time() - start_time)
    logging.info(f"Total time: {time.time() - start_time}")