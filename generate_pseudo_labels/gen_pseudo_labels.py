import numpy as np
import os
from tqdm import tqdm
import random
import operator
from tkinter import _flatten
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch

def buildDict_people(datalistFile, featsFile):      # Building data dictionary
    '''
    This method is to build data dictionary 
    for collecting positive and negative pair similarities
    '''
    imgName = []
    peopleName = []
    peopleList = set()
    with open(datalistFile, 'r') as f: datalist = f.readlines()
    for i in datalist: 
        tmpName = i.split()[0]
        tmpeopleName = tmpName.split('/')[-2]
        imgName.append(tmpName)
        peopleName.append(tmpeopleName)
        peopleList.add(tmpeopleName)
    feats = np.load(featsFile)
    peopleList = sorted(list(peopleList))
    featsDict = {}
    for index, value in enumerate(imgName):
        featsDict[value] = feats[index,:]
    print('='*20 + 'LOADING DATALIST' + '='*20)
    print(f"People Number: {len(peopleList)}")
    print(f"Image Number: {len(peopleName)}")
    print(f"Features Shape: {np.shape(feats)}")
    print('='*56)
    return featsDict, peopleList, peopleName, feats

def getIndex(target_value, data_list):       # Search
    '''
    Search the index of sample by identity
    '''   
    index = [i for i, v in enumerate(data_list) if v == target_value]
    return index

def cos(feats1, feats2):
    '''
    Computing cosine distance
    For similarity
    '''   
    cos = np.dot(feats1, feats2) / (np.linalg.norm(feats1) * np.linalg.norm(feats2))
    return cos

def gen_samepeople_Similarity(featsDict, peopleList, peopleName, feats, outfile_dist_info, fix_num=24):
    '''
    This method is to collect positive pair similarities
    '''
    print('='*20 + 'GENERATE POSITIVE PAIRS' + '='*20)
    imgsName = list(featsDict.keys())
    imgsame_list1 = []
    imgsame_list2 = []
    same_similaritys = []
    onepeople_count = 0
    id_similaritys = []
    pos_similarity_dist = {}
    for tar_people in tqdm(peopleList):
        index = getIndex(tar_people, peopleName)
        assert len(index) != 1
        id_similaritys = []
        for i in range(0,len(index)):
            compared_index = [value for value in index if index[i] != value]
            random.shuffle(compared_index)
            # avoid out_index
            pick_num = fix_num if len(compared_index) > fix_num else len(compared_index)            
            similaritys = []
            for j in range(pick_num):
                imgsame_list1.append(imgsName[index[i]])
                imgsame_list2.append(imgsName[compared_index[j]])
                embedding_similarity = cos(feats[index[i],:], feats[compared_index[j],:])
                same_similaritys.append(embedding_similarity)
                similaritys.append(embedding_similarity)
                id_similaritys.append(embedding_similarity)
            pos_similarity_dist[imgsName[index[i]]] = np.asarray(similaritys)
    same_similaritys = np.asarray(same_similaritys)
    assert len(imgsame_list1) == len(imgsame_list2) == len(same_similaritys)
    print(f"Positive Pair Number: {len(imgsame_list1)}")
    print(f"Maximum similarity value: {np.max(same_similaritys)}")
    print(f"Mean similarity value: {np.mean(same_similaritys)}")
    print(f"Minute similarity value: {np.min(same_similaritys)}")
    print(f"Std of similarity value: {np.std(same_similaritys)}")
    print('='*63)
    # Output the information of similarity
    outfile_dist_info.write('='*20 + " For Positive Pairs " + '='*20 + '\n')                       
    outfile_dist_info.write(f"One People Number: {onepeople_count}"+ '\n')
    outfile_dist_info.write(f"Positive Pair Number: {len(imgsame_list1)}"+ '\n')
    outfile_dist_info.write(f"Maximum similarity value: {np.max(same_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Mean similarity value: {np.mean(same_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Minute similarity value: {np.min(same_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Std of similarity value: {np.std(same_similaritys)}"+ '\n')
    return pos_similarity_dist, len(same_similaritys)

def gen_diffpeople_Similarity(featsDict, peopleList, peopleName, feats, allpospair_nums, outfile_dist_info, fix_num=24):
    '''
    This method is to collect negative pair similarities
    '''
    print('='*20 + 'GENERATE NEGATIVE PAIRS' + '='*20)
    imgsName = list(featsDict.keys())
    imgdiff_list1 = []
    imgdiff_list2 = []
    diff_similaritys = []
    id_similaritys = []
    onepeople_count = 0
    neg_similarity_dist = {}
    # pick_num = round(allpospair_nums / len(peopleName)) #mean pick num
    pick_num = fix_num
    compared_imgname = [value for value in imgsName]
    random.shuffle(compared_imgname)
    for tar_people in tqdm(peopleList):
        id_similaritys = []
        same_idx_list = getIndex(tar_people, peopleName)
        for i in same_idx_list:
            similaritys = []
            repetition = []
            pick_count = 0
            rand_num = 0
            while pick_count < pick_num:   # number of same people in negative pairs
                rand_num = random.randint(0,len(compared_imgname)-1)
                if compared_imgname[rand_num] in repetition: continue
                if compared_imgname[rand_num].split('/')[1] == imgsName[i].split('/')[1]: continue
                repetition.append(compared_imgname[rand_num])
                imgdiff_list1.append(imgsName[i])
                imgdiff_list2.append(compared_imgname[rand_num])
                embedding_similarity = cos(featsDict[imgsName[i]], featsDict[compared_imgname[rand_num]])
                diff_similaritys.append(embedding_similarity)
                similaritys.append(embedding_similarity)
                rand_num += 1
                pick_count +=1
                id_similaritys.append(embedding_similarity)
            neg_similarity_dist[imgsName[i]] = np.asarray(similaritys)
    diff_similaritys = np.asarray(diff_similaritys)
    assert len(imgdiff_list1) == len(imgdiff_list2) == len(diff_similaritys)
    print(f"One People Number: {onepeople_count}")
    print(f"Negative Pair Number: {len(imgdiff_list1)}")
    print(f"Maximum similarity value: {np.max(diff_similaritys)}")
    print(f"Mean similarity value: {np.mean(diff_similaritys)}")
    print(f"Minute similarity value: {np.min(diff_similaritys)}")
    print(f"Std of similarity value: {np.std(diff_similaritys)}")
    print('='*63)
    outfile_dist_info.write('='*20 + " For Negative Pairs " + '='*20 + '\n')
    outfile_dist_info.write(f"One People Number: {onepeople_count}"+ '\n')
    outfile_dist_info.write(f"Negative Pair Number: {len(imgdiff_list1)}"+ '\n')
    outfile_dist_info.write(f"Maximum similarity value: {np.max(diff_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Mean similarity value: {np.mean(diff_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Minute similarity value: {np.min(diff_similaritys)}"+ '\n')
    outfile_dist_info.write(f"Std of similarity value: {np.std(diff_similaritys)}"+ '\n')
    return neg_similarity_dist

def gen_distance(pos_similarity_dist, neg_similarity_dist, outfile):
    '''
    This method is to calculate quality scores via wasserstein distance
    '''
    outfile = open(outfile, 'w')
    print('='*20 + 'OUTPUT' + '='*20)
    posimglist = list(pos_similarity_dist.keys())
    negimglist = list(pos_similarity_dist.keys())
    assert len(posimglist) == len(negimglist)
    imglists = posimglist
    for img in tqdm(imglists):
        tar_pos_similaritys = pos_similarity_dist[img]
        tar_neg_similaritys = neg_similarity_dist[img]
        w_distance = wasserstein_distance(tar_pos_similaritys, tar_neg_similaritys)
        outfile.write(img + '\t' + str(w_distance) + '\n')

def cal_idscore(result_file):
    '''
    This method is to calculate quality scores of identity
    '''
    with open(result_file, 'r') as f: txtContent = f.readlines()
    peoid= []
    quality_scores = []
    idsocre_dist = {}
    for i in txtContent: idsocre_dist[i.split()[0].split('/')[1]] = [0,0]  # init
    quality_scores = []
    for i in txtContent: quality_scores.append(float(i.split()[1]))
    # normalize quality pseudo labels
    quality_scores = (quality_scores - np.min(quality_scores)) / \
                     (np.max(quality_scores) - np.min(quality_scores)) * 100
    count = 0
    peoid = set()
    for i in tqdm(txtContent):
        idname = i.split()[0].split('/')[1]
        idsocre_dist[idname][0] += quality_scores[count]
        idsocre_dist[idname][1] += 1
        count += 1
        peoid.add(idname)
    peoid = list(peoid)
    id_score = {}
    for i in tqdm(peoid): id_score[i] = (idsocre_dist[i][0] / idsocre_dist[i][1])
    return id_score
 
def norm_labels(data_root, outfile_wdistacne, outfile_result, id_score):
    '''
    This method is to normalize quality scores
    '''
    outfile_result = open(outfile_result, 'w')
    with open(outfile_wdistacne, 'r') as f: txtContent = f.readlines()
    imgpath = []
    quality_scores = []
    for i in tqdm(txtContent):
        imgname = i.split()[0]
        mean_wdistance = float(i.split()[1])
        imgpath.append(data_root + imgname)
        quality_scores.append(mean_wdistance)
    quality_scores = np.asarray(quality_scores)
    quality_scores = (quality_scores-np.min(quality_scores)) / \
                     (np.max(quality_scores) - np.min(quality_scores)) * 100
    quality_scores_select = []
    imgpath_select = []
    for i in tqdm(range(len(quality_scores))):
        quality_scores_select.append(quality_scores[i])
        imgpath_select.append(imgpath[i])
    quality_scores_select = (quality_scores_select-np.min(quality_scores_select)) / \
                            (np.max(quality_scores_select) - np.min(quality_scores_select)) * 100
    return imgpath_select, quality_scores_select

if __name__ == "__main__":
    '''
    This method is to generate quality pseudo scores by similarity distribution distance (SDD)
    and save it to txt files
    '''
    data_root    = '/data/home/fuzhaoou/Dataset/MS1M/MS1M_arcface_112'
    datalistFile = './DATA.label'
    featsFile    = './feats_npy/Embedding_Features.npy'
    create_dir   = './annotations'
    os.makedirs(create_dir, exist_ok=True)
    outfile_dist_info = f"{create_dir}/distribution_info_tmp.txt"
    outfile_wdistacne = f"{create_dir}/w_distances.txt"
    outfile_result    = f"{create_dir}/quality_pseudo_labels.txt"
    outfile_dist_info = open(outfile_dist_info, 'w')
    featsDict, peopleList, peopleName, feats = buildDict_people(datalistFile,featsFile)
    quality_scores_metrix = np.zeros([len(peopleName), 12])
    for i in range(12):
        pos_similarity_dist, pos_pairs_num = gen_samepeople_Similarity(featsDict, peopleList, \
                                                        peopleName, feats, outfile_dist_info)
        neg_similarity_dist = gen_diffpeople_Similarity(featsDict, peopleList, peopleName, \
                                                        feats, pos_pairs_num, outfile_dist_info)
        gen_distance(pos_similarity_dist, neg_similarity_dist, outfile_wdistacne)
        id_score = cal_idscore(outfile_wdistacne)
        imgpath_select, quality_scores_select = norm_labels(data_root, outfile_wdistacne, \
                                                            outfile_result, id_score)
        quality_scores_metrix[:,i] = quality_scores_select
    quality_pseudo_labels = np.mean(quality_scores_metrix, axis=1)
    outfile_result = open(outfile_result, 'w')
    # output quality pseudo labels
    for index, value in enumerate(quality_pseudo_labels):                      
        outfile_result.write(imgpath_select[index] + '\t' + str(value) + '\n')
