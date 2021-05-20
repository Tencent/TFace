## Introduction
This is the official implement of CVPR2021 "SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance" on Pytorch.

## DataSet
**Training Datasets for SDD-FIQA**
`MS1MV2` `VGGFace2`

**Testing Datasets for SDD-FIQA**，
`LFW` `Adience` `IJB-C` 

## Requirements 
The pretrained basic face recognition model


## Generate quality pseudo label
2. Run './generate_pseudo_labels/gen_datalist.py' to generate pair list.
3. Run './generate_pseudo_labels/extract_embedding/extract_feats.py' to obtain recognition features.
4. Run './generate_pseudo_labels/gen_pseudo_labels.py' to calculate quality pseudo-labels.


## Local Training of  Quality Regression Model
1. Replace the data path with your loacl path on `train_confing.py`.
2. bash local_train.sh.

## Testing of Quality Regression Model 
Run './eval.py' to predict face quality score.
We provide the pretrained model on MS1MV2 with IR50： [googledrive](https://drive.google.com/file/d/1AM0iWVfSVWRjCriwZZ3FXiUGbcDzkF25/view?usp=sharing)

## Citing this repository
If you find this code useful in your research, please consider citing us:
```
@article{sdd2021,
  title={SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance},
  author={Ou, Fu-Zhao and Chen, Xingyu and Zhang, Ruixin and Huang, Yuge and Li, Shaoxin and Li, Jilin and Li, Yong and Cao, Liujuan and Wang, Yuan-Gen},
  journal={arXiv preprint arXiv:2103.05977},
  year={2021}
}
```