# SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance

## Introduction
####
In recent years, Face Image Quality Assessment (FIQA) has become an indispensable part of the face recognition system to guarantee the stability and reliability of recognition performance in an unconstrained scenario. In this work, we argue that a high-quality face image should be similar to its intra-class samples and dissimilar to its inter-class samples. Thus, we propose a novel unsupervised FIQA method that incorporates Similarity Distribution Distance for Face Image Quality Assessment (SDD-FIQA). Our method generates quality pseudo-labels by calculating the Wasserstein Distance (WD) between the intra-class and inter-class similarity distributions. With these quality pseudo-labels, we are capable of training a regression network for quality prediction. Extensive experiments on benchmark datasets demonstrate that the proposed SDD-FIQA surpasses the state-of-the-arts by an impressive margin. Meanwhile, our method shows good generalization across different recognition systems. 
<img src="docs/framework.png" title="SDD-FIQA framework" width="500" />


## Generation of Quality Pseudo-Labels
2. Run './generate_pseudo_labels/gen_datalist.py' to obtain the data list.
3. Run './generate_pseudo_labels/extract_embedding/extract_feats.py' to extract the embeddings of face image.
4. Run './generate_pseudo_labels/gen_pseudo_labels.py' to calculate the quality pseudo-labels.


## Training of Quality Regression Model
1. Replace the data path with your local path on `train_confing.py`.
2. Run local_train.sh.

## Prediction of FIQA 
Run './eval.py' to predict face quality score.
We provide the pre-trained model on the refined MS1M dataset with IR50: [googledrive](https://drive.google.com/file/d/1AM0iWVfSVWRjCriwZZ3FXiUGbcDzkF25/view?usp=sharing)

## Results
<img src="docs/res.png" title="results" width="500" />

## Citing this Repository
If you find this code useful in your research, please consider citing us:
```
@InProceedings{SDD-FIQA2021,
   author={Ou, Fu-Zhao and Chen, Xingyu and Zhang, Ruixin and Huang, Yuge and Li, Shaoxin and Li, Jilin and Li, Yong and Cao, Liujuan and Wang, Yuan-Gen},
   title = {{SDD-FIQA}: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance},
   booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
   year = {2021},
}

```
