#/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import sys
sys.path.insert(0, "/home/fuxueping/sdb/Caffe_Project_Train/caffe-ssd/python")
import caffe
import time
import numpy as np
import os
import os.path
import copy
import cv2
from comfunc import *
import matplotlib.pyplot as plt

###sphereface
# deploy='/home/fuxueping/sdb/PycharmProjects/PytorchToCaffe/models/resnet27/seqface/deploy.prototxt'
# caffe_model='/home/fuxueping/sdb/PycharmProjects/PytorchToCaffe/models/resnet27/seqface/train.caffemodel'
deploy='/home/fuxueping/sdb/PycharmProjects/PytorchToCaffe/models/deploy_mgbn.prototxt'
caffe_model='/home/fuxueping/sdb/PycharmProjects/PytorchToCaffe/models/resnet20_mgbn.caffemodel'


batch =20
fpr_TH_000 = 0.001
fpr_TH_0000 = 0.0001
# imgSize = [128,128] # [h,w]
imgSize = [112,112] # [h,w]
steps = 100
step = 1.0 / steps
scale = 0.0078125
mean_value = 127.5
threshold_same = 0.5
threshold_diff = 0.65
featureLenth = 512


def initilizeNet():
    print ('initilize ... ')
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(deploy, caffe_model, caffe.TEST)
    return net

def extractFeature(net,ListImgPath,batch):
    Num = len(ListImgPath)
    iteration = Num // batch
    left = Num % batch
    listFeatures = []
    for i in range(iteration + 1):
        if i == iteration:
            batch_size = left
            if left == 0:
                return listFeatures
        else:
            batch_size = batch
        tempimg = np.zeros((batch_size, imgSize[0], imgSize[1], 3))
        for j in range(batch_size):
            index = i*batch+j
            imgpath = ListImgPath[index].split()[0]
            # imgpath = '/home/fuxueping/sdb/Caffe_Project/face_recognition/train/faces_ms1m_112x112/imgs_part/1/116.jpg'
            im = cv2.imread(imgpath)
            im = cv2.resize(im, (112, 112))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        #after mxnet2caffe model
            if im is None:
                print("image is null.check path:%s"%imgpath)
                exit(0)
            tempimg[j , :, :, :] = im
        tempimg = (tempimg - mean_value) * scale  # done in imResample function wrapped by python
        tempimg = tempimg.transpose(0, 3, 1, 2)
        # print(tempimg)
        # print("end")
        # start1 = time.clock()
        net.blobs['data'].reshape(batch_size, 3, imgSize[0], imgSize[1])
        net.blobs['data'].data[...] = tempimg
        net.reshape()
        net.forward()
        feat = net.blobs['fc1'].data
        # end1 = time.clock()
        for r in range(feat.shape[0]):
            batch_feat = list(feat[r])
            l2_norm = cv2.norm(np.array(batch_feat), cv2.NORM_L2)
            if l2_norm > 0:
                batch_feat = list(np.array(batch_feat) / l2_norm)
            else:
                batch_feat = np.zeros((1, featureLenth))
            listFeatures.append(batch_feat)
        # print("extractFeature time is : %.04f seconds" % (end1 - start1))

    return listFeatures

def traversalDir_FirstDir(path):
    list = []
    listID = []
    ListImgPath = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path, file)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                list.append(h[1])
                PerImgsPath = os.path.join(path,h[1])
                if (os.path.exists(PerImgsPath)):
                     imgs = os.listdir(PerImgsPath)
                     for img in imgs:
                         pathFull = PerImgsPath + '/' + img
                         im = cv2.imread(pathFull)
                         im = cv2.resize(im, (112, 112))
                         if im.shape[0] != imgSize[0]:
                             continue
                         ID = pathFull.split('/')[-2].split('G')[0]
                         # ID = ID[::-1]
                         listID.append(ID)
                         ListImgPath.append(pathFull)

    return ListImgPath, listID



def test_id_verification(regfeat,capfeat,regid,capid,regimg,capimg,resultTxt):

    re_fpr, re_tpr, re_acc, min_fpr,min_fpr_001,min_fpr_0001 = [], [], [], [],[],[]
    f_re = open(resultTxt, 'w')
    # f_same = open('/media/algorithm-5/cb20b39d-1099-435b-a52c-8e57e5baeaac/Maxvision_faceData/testResult/1022/verification/same_err.txt','w')
    # f_diff = open('/media/algorithm-5/cb20b39d-1099-435b-a52c-8e57e5baeaac/Maxvision_faceData/testResult/1022/verification/diff_err.txt','w')

    regmat = np.array(regfeat)
    capmat = np.array(capfeat)

    regmat_T = regmat.T
    SimilarityMatrix = np.dot(capmat, regmat_T)
    sameListscore = []
    diffListscore = []
    for i in range(len(capid)):
        for j in range(len(regid)):
            if capid[i] == regid[j]:
                sameListscore.append(SimilarityMatrix[i, j])
                # if SimilarityMatrix[i, j] <= threshold_same:
                    # f_same.writelines(capimg[i] + ' ' + regimg[j] + ' ' + str(SimilarityMatrix[i, j]) + '\n')

            else:
                diffListscore.append(SimilarityMatrix[i, j])
                # if SimilarityMatrix[i, j] >= threshold_diff:
                    # f_diff.writelines(capimg[i] + ' ' + regimg[j] + ' ' + str(SimilarityMatrix[i, j]) + '\n')

    sameNum = len(sameListscore)
    diffNum = len(diffListscore)
    ratio = sameNum / diffNum

    sigma = 0
    s1 = np.array(sameListscore)
    s2 = np.array(diffListscore)
    for s in range(steps):
        sigma += step   # step = 0.01

        s1[s1 < sigma] = 0      #s1 sameListscore相同对得分列表
        TPnum = np.count_nonzero(s1)    #同一人且高于阈值，正确识别
        FNnum = sameNum - TPnum     #同一人低于阈值,拒识

        s2[s2 < sigma] = 0      ##s2 diffListscore不同对得分列表
        FPnum = np.count_nonzero(s2)     #不同人高于阈值，误认
        TNnum = diffNum - FPnum     #不同人低于阈值

        if TPnum + FNnum != 0 and FPnum + TNnum != 0:
            fpr = FPnum * 1.0 / (TNnum + FPnum)
            tpr = TPnum * 1.0 / (TPnum + FNnum)
            acc = (TPnum + TNnum * ratio) / ((TPnum + FNnum) + (TNnum + FPnum) * ratio)
            re_fpr.append(fpr)
            re_tpr.append(tpr)
            re_acc.append(acc)
            min_fpr_001.append(abs(fpr - fpr_TH_000))
            min_fpr_0001.append(abs(fpr - fpr_TH_0000))
            reStr = "sigma= " + str(sigma) + " , fpr= " + str(fpr) + " , tpr= " + str(tpr) + " , acc= " + str(
                acc) + " , FP= " + str(FPnum) + " , TP= " + str(TPnum) + \
                    ', FP+FN=' + str(FPnum + sameNum - TPnum)
            print(reStr)
            f_re.writelines(reStr + '\n')
        else:
            print ("result file is wrong")
            exit(-1)

    final_001_fpr_index = min_fpr_001.index(min(min_fpr_001))
    final_001_fpr = re_fpr[final_001_fpr_index]
    final_sigma_001 = final_001_fpr_index * step + step
    final_001_tpr = re_tpr[final_001_fpr_index]

    final_0001_fpr_index = min_fpr_0001.index(min(min_fpr_0001))
    final_0001_fpr = re_fpr[final_0001_fpr_index]
    final_sigma_0001 = final_0001_fpr_index * step + step
    final_0001_tpr = re_tpr[final_0001_fpr_index]

    final_zero_fpr_index = re_fpr.index(0, 1)
    final_zero_sigma = final_zero_fpr_index * step + step
    final_zero_tpr = re_tpr[final_zero_fpr_index]

    print('\n')
    reStr = "same pairs = " + str(sameNum) + " , diff pairs = " + str(diffNum) + " ,  ratio = " + str(ratio)
    print(reStr)
    f_re.writelines(reStr + '\n')
    reStr = 'The fpr that you care : ' + str(final_001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_001) + '   and the corresponding tpr : ' + str(final_001_tpr) + '\n'
    f_re.writelines(reStr + '\n')
    print(reStr)

    reStr = 'The fpr that you care : ' + str(final_0001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_0001) + '   and the corresponding tpr : ' + str(final_0001_tpr) + '\n' +\
            'The fpr that you care : ' + str(0) + '   and the corresponding sigma : ' + str(final_zero_sigma) + '   and the corresponding tpr : ' + str(final_zero_tpr)

    f_re.writelines(reStr + '\n')
    print(reStr)

    return s1, s2


def test_top1_identification(regfeat,capfeat,regid,capid,resultTxt):

    re_fpr, re_tpr, min_fpr,min_fpr_001,min_fpr_0001 = [], [], [],[],[]
    f_re = open(resultTxt, 'w')
    # f_same = open('/media/algorithm-5/cb20b39d-1099-435b-a52c-8e57e5baeaac/Maxvision_faceData/same_err.txt', 'w')
    # f_diff = open('/media/algorithm-5/cb20b39d-1099-435b-a52c-8e57e5baeaac/Maxvision_faceData/diff_err.txt', 'w')

    regmat = np.array(regfeat)
    capmat = np.array(capfeat)

    regmat_T = regmat.T
    SimilarityMatrix = np.dot(capmat, regmat_T)
    sameListscore = []
    diffListscore = []

    top_id_List = []
    top_score_List = []
    top_im_List = []
    cap_id_List = capid
    rec_id_List = regid
    # rec_im_List = regimg
    # cap_im_List = capimg
    #  去重  #
    for i in range(len(capid)):
        for j in range(len(regid)):
            if capid[i] == regid[j]:
                sameListscore.append(SimilarityMatrix[i, j])
                # if SimilarityMatrix[i, j] <= threshold_same:
                #     f_same.writelines(capimg[i] + ' ' + regimg[j] + ' ' + str(SimilarityMatrix[i, j]) + '\n')
            else:
                diffListscore.append(SimilarityMatrix[i, j])
                # if SimilarityMatrix[i, j] >= threshold_diff:
                #     f_diff.writelines(capimg[i] + ' ' + regimg[j] + ' ' + str(SimilarityMatrix[i, j]) + '\n')

    sameNum = len(sameListscore)
    diffNum = len(diffListscore)
    ratio = sameNum / diffNum
    print('\n')
    reStr = "same pairs = " + str(sameNum) + " , diff pairs = " + str(diffNum) + " ,  ratio = " + str(ratio)
    print(reStr)
    f_re.writelines(reStr + '\n')

    for i in range(len(capid)):
        rec_score_List = list(SimilarityMatrix[i, :])
        top_score = max(rec_score_List)
        top_idx = rec_score_List.index(top_score)
        top_id = rec_id_List[top_idx]
        # top_im = rec_im_List[top_idx]
        top_id_List.append(top_id)
        top_score_List.append(top_score)
        # top_im_List.append(top_im)

    in_id_list = [x for x in capid if x in regid]
    in_num = len(in_id_list)
    total_num = len(capid)

    sigma = 0
    for s in range(steps):
        sigma += step  # step = 0.01
        FPNum, TPNum = 0, 0
        for j in range(len(capid)):
            if cap_id_List[j] != top_id_List[j] and top_score_List[j] > sigma:
                FPNum +=1
                # im_pair = (cap_im_List[i],top_im_List[i])
            elif cap_id_List[j] == top_id_List[j] and top_score_List[j]> sigma:
                TPNum +=1


        fpr = FPNum/total_num
        tpr = TPNum/in_num
        min_fpr_001.append(abs(fpr - fpr_TH_000))
        min_fpr_0001.append(abs(fpr - fpr_TH_0000))
        re_fpr.append(fpr)
        re_tpr.append(tpr)
        reStr = "sigma= " + str(sigma) + " , FPNum= " + str(FPNum)+ " , TPNum= " + str(TPNum)+ " , FPR= " + str(fpr) + " , TPR= " + str(tpr)
        print (reStr)
        f_re.writelines(reStr + '\n')

    final_001_fpr_index = min_fpr_001.index(min(min_fpr_001))
    final_001_fpr = re_fpr[final_001_fpr_index]
    final_001_tpr = re_tpr[final_001_fpr_index]
    final_sigma_001 = final_001_fpr_index * step + step

    final_0001_fpr_index = min_fpr_0001.index(min(min_fpr_0001))
    final_0001_fpr = re_fpr[final_0001_fpr_index]
    final_0001_tpr = re_tpr[final_0001_fpr_index]
    final_sigma_0001 = final_0001_fpr_index * step + step

    final_zero_fpr_index = re_fpr.index(0, 1)
    final_zero_sigma = final_zero_fpr_index * step + step
    final_zero_tpr = re_tpr[final_zero_fpr_index]

    print("all cap num:%d,cap in reg num:%d\n" % (total_num, in_num))
    reStr = 'The fpr that you care : ' + str(final_001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_001) + '   and the corresponding tpr : ' + str(final_001_tpr) + '\n'
    print (reStr)
    f_re.writelines(reStr + '\n')

    reStr = 'The fpr that you care : ' + str(final_0001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_0001) + '   and the corresponding tpr : ' + str(final_0001_tpr) + '\n'+\
            'The fpr that you care : ' + str(0) + '   and the corresponding sigma : ' + str(final_zero_sigma) + '   and the corresponding tpr : ' + str(final_zero_tpr)
    print (reStr)
    f_re.writelines(reStr + '\n')

def Score_analyse(same_score, diff_score):
    s_mean = np.mean(same_score)
    d_mean = np.mean(diff_score)
    s_var = np.var(same_score)
    d_var = np.var(diff_score)
    S_max = max(same_score)
    S_min = min(same_score)
    D_max = max(diff_score)
    D_min = min(diff_score)
    [S_score, S_x] = np.histogram(same_score, bins=100, range=(0, 1))
    [D_score, D_x] = np.histogram(diff_score, bins=100, range=(0, 1))
    ## print("S_max: ",S_max,"    S_min: ",S_min,"    ;     D_max: ",D_max,"    D_min: ",D_min)
    ## print("s_mean: ", s_mean, "    s_var: ", s_var, "    ;     d_mean: ", d_mean, "    d_var: ", d_var)
    plt.hist(same_score, bins=100, density = 1, facecolor="red", edgecolor="black", label='same', alpha=0.7, hold = 1)
    plt.hist(diff_score, bins=100, density = 1, facecolor="blue", edgecolor="black",label='different', alpha=0.7)
    # plt.hist(same_score, bins=100, facecolor="red", edgecolor="black", label='same', alpha=0.7)
    # plt.hist(diff_score, bins=100, facecolor="blue", edgecolor="black",label='different', alpha=0.7)
    plt.title("same/different pairs score histogram of model 29")
    plt.xlabel("score")
    plt.ylabel('pairs')
    plt.yticks([])
    plt.legend()
    plt.show()


def write_same_and_differernt(jsonfile):
    f = open(jsonfile, 'r')
    lines = f.read()
    jsonobj = json.loads(lines)
    same_scores = []
    diff_scores = []
    for i in range(len(jsonobj["RECORDS"])):
        score=jsonobj["RECORDS"][i]["result"]
        if jsonobj["RECORDS"][i]["same_name"]=="1":
            same_scores.append(score)
        else:
            diff_scores.append(score)
    return same_scores, diff_scores


#错误识别对分别拷贝到文件夹便于核查
def copy_pairlist(file,pairdir):

    fp = open(file,'r')   #file记录不同文件夹内相似度大于0.85
    list = fp.readlines()
    num = 0
    for line in list:
       imgpath_1 = line.strip().split(' ')[0]
       imgname_1 = imgpath_1.split('/')[-1]
       imgpath_2 = line.strip().split(' ')[1]
       imgname_2 = imgpath_2.split('/')[-1]
       score = line.strip().split(' ')[2]
       foldname = str(num) + '_' + str(score)
       newfold = os.path.join(pairdir,foldname)
       if not os.path.exists(newfold):
           os.mkdir(newfold)
       newfile_1 = newfold + '/' +imgname_1
       newfile_2 = newfold + '/' +imgname_2
       shutil.copy(imgpath_1,newfile_1)
       shutil.copy(imgpath_2,newfile_2)
       num = num +1
       print(foldname+'\n')
    pass
    print('fininshed')




def test(meth):

    net = initilizeNet()
    resultPath = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/result/{}/sphere_res50.txt'.format(meth)

    # pairpath = '/home/nyy/DataSets/FPpairs_0.60'
    # file = 'diff_err.txt'

    if meth == 'identification':
        # regbase ='/home/fuxueping/sdb/Caffe_Project/face_recognition/test/identification_{}_{}/reg'.format(imgSize[0],imgSize[1])
        # capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/identification_{}_{}/cap'.format(imgSize[0],imgSize[1])

        regbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/reg'
        capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/cap'


        regimglist, regidlist = traversalDir_FirstDir(regbase)
        regfeat = extractFeature(net, regimglist, batch)

        capimglist, capidlist = traversalDir_FirstDir(capbase)
        capfeat = extractFeature(net, capimglist, batch)

        test_top1_identification(regfeat, capfeat, regidlist, capidlist, resultPath)


    else:

        # regbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/verification_{}_{}/TestID/ID'.format(
        #     imgSize[0], imgSize[1])
        # capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/verification_{}_{}/TestID/face'.format(
        #     imgSize[0], imgSize[1])

        regbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/reg'
        capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/cap'

        regimglist, regidlist = traversalDir_FirstDir(regbase)
        regfeat = extractFeature(net, regimglist, batch)

        capimglist, capidlist = traversalDir_FirstDir(capbase)
        capfeat = extractFeature(net, capimglist, batch)

        same_score, diff_score = test_id_verification(regfeat,capfeat,regidlist,capidlist,regimglist,
                          capimglist,resultPath)

        #Score_analyse(same_score, diff_score)
        # copy_pairlist(file,pairpath)


if __name__ == "__main__":

    method = 'identification'#识别
    # method = 'verification'#验证
    test(method)
