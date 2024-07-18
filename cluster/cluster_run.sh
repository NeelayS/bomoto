ARG=$1
EXP_NAME=$2
python /is/cluster/fast/sbhor/bomoto/run.py --cfg /is/cluster/fast/sbhor/bomoto/configs/params_dataset.yaml \
--cluster_batch_size=56 --cluster_start_idx=237 --input-dir /is/cluster/sbhor/smpl_ground_truth_corr/ --output-dir /is/cluster/fast/sbhor/star_bedlam_bomoto/ #--object_key=0
