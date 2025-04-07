### Verification test

LFW, CFP-FP, CPLFW, AgeDB, CALFW are popular verification datasets in face recognition task, below is test steps.

1. Download the bin files and save into `val_data_dir`, the test checkpoint path is `ckpt_path`.
2. Run test codes：
``` bash
export CUDA_VISIBLE_DEVICES='0'
python -u verification.py --ckpt_path=$ckpt_path --data_root=$val_data_dir
```

RFW is the common test datasets for fairness, the test code is `verification_rfw.py`

### 1:1 test

IJB-B and IJB-C are most common large-scale face 1:1 test protocols.
1. Download the raw image data and meta files, saved into `data_root`
2. Run test codes：
``` bash
# extract face features
export CUDA_VISIBLE_DEVICES='0,1,2,3'
python -u extract_features.py  --ckpt_path=$ckpt_path --backbone=$backbone_name --gpu_ids=$gpu_ids \
                                --batch_size=512 --data_root=$data_root \
                                --filename_list=$pts_score_file \
                                --output_dir=$output_dir
# evaluation
python -u IJB_Evaluation.py --dataset=$dataset --meta_dir=$meta_dir \
                              --feature="${output_dir}"/"feature.npy" --face_scores=${output_dir}/"faceness_scores.npy" \
                              --output_name=${output_dir}/"similarity.npy"

```