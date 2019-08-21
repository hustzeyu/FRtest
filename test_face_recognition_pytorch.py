#/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
# import sys
# sys.path.insert(0,'/home/fuxueping/sdb/Caffe_Project_Train/caffe-sphereface/python')
# import caffe
import time
import numpy as np
import os
import os.path
import copy
import cv2
from comfunc import *
import matplotlib.pyplot as plt
from PIL import Image

import torch
from model import MobileFaceNet
from torchvision import transforms
from resnet import resnet101
# from model_irse import IR_SE_50,Backbone
from model import Backbone
# from model_seresnet import Resnet50
from models.SEresnet50 import Resnet50
from models.Resnet20 import Resnet20
###mobileFaceNet
# model = ''
# model='/home/fuxueping/sdb/Caffe_Project_Train/github/work_space_epoch30/save/model_2019-06-11-07-41_accuracy:0.8937142857142856_step:7500_final.pth'
# model = '/home/fuxueping/sdb/Caffe_Project_Train/github/work_space_epoch30_triplet900000/save/model_2019-06-13-12-05_accuracy:0.8914285714285715_step:75000_final.pth'
# model = '/home/fuxueping/sdb/Caffe_Project_Train/github/work_space_epoch50_triplets900000_distill/save/model_2019-06-21-07-35_accuracy:0.8507142857142856_step:125000_final.pth'
# model = '/home/fuxueping/sdb/Caffe_Project_Train/github/work_space_epoch50_triplets900000_mobileFace/save/model_2019-06-21-16-37_accuracy:0.8028571428571428_step:231160_final.pth'
# model = '/home/fuxueping/sdb/Caffe_Project_Train/github/work_space_epoch75_msair6_mobileFace/save/model_2019-07-01-15-59_accuracy:0.8388571428571427_step:1443975_final.pth'
# model = '/home/fuxueping/sdb/Caffe_Project_Train/github/work_space/models/model_2019-08-01-02-44_accuracy:0.8045714285714286_step:47003_None.pth'
##resnet101
# model = '/home/fuxueping/sdb/Caffe_Project_Train/face_recognition/resnet100/resnet101-5d3b4d8f.pth'

#resnet50
# model = '/home/fuxueping/sdb/Caffe_Project_Train/github/work_spzce_epoch75_msair6_SEResenet50/save/123.pth'
# model = '/home/fuxueping/sdb/PycharmProjects/caffeToPytorch/result/SEresnet50_model.pth'

#resnet20
model = '/home/fuxueping/sdb/Caffe_Project_Train/github/temp/model_2019-08-17-08-13_accuracy_0.8282857142857143_step_485128_None_epoch_11.pth'


batch =20
fpr_TH =0.0001
# imgSize = [128,128] # [h,w]
imgSize = [112, 112] # [h,w]
steps = 100
step = 1.0 / steps
scale = 0.0078125
mean_value = 127.5
threshold_same = 0.5
threshold_diff = 0.65
featureLenth = 512


def initilizeNet_mobileFace():
    print ('initilize ... ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = MobileFaceNet(512).to(device)
    return net


def initilizeNet():
    print('initilize ... ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net = IR_SE_50(imgSize).to(device)
    # net = Backbone(50, 0.6, 'ir_se').to(device)#pytorch自己训练的版本
    # net = Resnet50(512).to(device)#mxnet版本
    net = Resnet20(embedding_size = 512).to(device)
    backbone_dict = net.state_dict()
    pretrained_dict = torch.load(model)
    pretrained_dict_backbone_ = {}
    for k, v in pretrained_dict.items():
        k_ = k.replace('module.', '')
        if k_ in backbone_dict and backbone_dict[k_].size() == v.size():
            pretrained_dict_backbone_[k_] = v
        else:
            print(k_, " is not in backbone_dict!")

    backbone_dict.update(pretrained_dict_backbone_)
    net.load_state_dict(backbone_dict)
    return net

def initilizeNet_res101():
    print ('initilize ... ')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = resnet101()
    # net.load_state_dict(torch.load(model))
    checkpoint = torch.load(model)
    # print(checkpoint)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # state_dict = {}
    # for k, v in checkpoint.items():
    #     k_ = k.replace("module.", "")
    #     # k_ = 'feat_extract.'+k
    #     if k_ == 'fc':
    #         continue
    #     else:
    #         state_dict[k_] = v

    net.load_state_dict(state_dict)
    net.to(device)
    return net

def load_image(img_path):
    # image = cv2.imread(img_path, 0)
    image = cv2.imread(img_path)
    # image = cv2.resize(image, (112, 112))
    if image is None:
        return None
    # image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    # image = image[:, np.newaxis, :, :]
    image = image[np.newaxis, :, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def get_features_resnet50(net, ListImgPath,Batch):
    net.eval()

    Num = len(ListImgPath)
    iteration = Num // Batch
    left = Num % Batch
    listFeatures = []
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    for i in range(iteration + 1):
        if i == iteration:
            batch_size = left
            if left == 0:
                return listFeatures
        else:
            batch_size = Batch
        tempimg = torch.FloatTensor(batch_size, 3, imgSize[0], imgSize[1])
        for j in range(batch_size):
            index = i * Batch + j
            imgpath = ListImgPath[index].split()[0]
            im = cv2.imread(imgpath)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im is None:
                print("image is null.check path:%s" % imgpath)
                exit(0)

            im_tensor = test_transform(im).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).unsqueeze(
                0)
            tempimg[j, :, :, :] = im_tensor

        with torch.no_grad():
            # features = l2_norm(model(batch.to(device))).cpu().detach().numpy()
            # batch_feat_tensor = torch.nn.functional.normalize(net(tempimg.cuda()).reshape(tempimg.shape[0], 512))
            batch_feat_tensor = net(tempimg.cuda()).reshape(tempimg.shape[0], 512)
        batch_feat_numpy = batch_feat_tensor.cpu().data.numpy()
        # print("im_tensor", batch_feat_numpy)
        for every in batch_feat_numpy:
            listFeatures.append(every.tolist())
            # print("img_feat", every.tolist())

    print(len(listFeatures))
    return listFeatures

def get_features_SEresnet50( net,ListImgPath,Batch):
    net.eval()
    Num = len(ListImgPath)
    iteration = Num // Batch
    left = Num % Batch
    listFeatures = []
    for i in range(iteration + 1):
        if i == iteration:
            batch_size = left
            if left == 0:
                return listFeatures
        else:
            batch_size = Batch
        tempimg = torch.FloatTensor(batch_size, 3, imgSize[0], imgSize[1])
        for j in range(batch_size):
            index = i * Batch + j
            imgpath = ListImgPath[index].split()[0]
            image = load_image(imgpath)
            if image is None:
                print("image is null.check path:%s" % imgpath)
                exit(0)
            tempimg[j, :, :, :] = torch.from_numpy(image)
            with torch.no_grad():
                batch_feat_tensor1 = net(tempimg.cuda())
                batch_feat_tensor = batch_feat_tensor1.reshape(tempimg.shape[0], 512)
            batch_feat_numpy = batch_feat_tensor.cpu().data.numpy()
            # print("im_tensor", batch_feat_numpy)
            for every in batch_feat_numpy:
                listFeatures.append(every.tolist())
                # print("img_feat", every.tolist())

    print(len(listFeatures))
    return listFeatures


def extractFeature_mobileFace( net,ListImgPath,Batch):

    # myresnet.load_state_dict(torch.load(args.model))
    #gpu 0
    # checkpoint = torch.load(model,map_location=lambda storage, loc: storage.cuda(0))
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    # net.load_state_dict(state_dict)
    #other gpu

    # net.load_state_dict(torch.load(model, map_location={'cuda:4,5,6,7': 'cuda:0'}))

    net.eval()

    Num = len(ListImgPath)
    iteration = Num // Batch
    left = Num % Batch
    listFeatures = []
    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # range [0.0, 1.0] -> [-1.0,1.0]
    for i in range(iteration + 1):
        if i == iteration:
            batch_size = left
            if left == 0:
                return listFeatures
        else:
            batch_size = Batch
        tempimg = torch.FloatTensor(batch_size, 3, imgSize[0], imgSize[1])
        for j in range(batch_size):
            index = i * Batch + j
            imgpath = ListImgPath[index].split()[0]
            im = cv2.imread(imgpath)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if im is None:
                print("image is null.check path:%s" % imgpath)
                exit(0)

            im_tensor = test_transform(im).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).unsqueeze(0)
            tempimg[j, :, :, :] = im_tensor

        with torch.no_grad():
            batch_feat_tensor = net(tempimg.cuda()).reshape(tempimg.shape[0], 512)
        batch_feat_numpy = batch_feat_tensor.cpu().data.numpy()
        # print("im_tensor", batch_feat_numpy)
        for every in batch_feat_numpy:
            listFeatures.append(every.tolist())
                # print("img_feat", every.tolist())

    print(len(listFeatures))
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
                         if im.shape[0] != imgSize[0]:
                             continue
                         ID = pathFull.split('/')[-2]
                         listID.append(ID)
                         ListImgPath.append(pathFull)

    return ListImgPath,listID


def test_id_verification(regfeat,capfeat,regid,capid,regimg,capimg,resultTxt):

    re_fpr, re_tpr, re_acc, min_fpr = [], [], [], []
    f_re = open(resultTxt, 'w')
    f_same = open('/home/fuxueping/sdb/Caffe_Project/face_recognition/test/result/same_err_mobileface.txt','w')
    f_diff = open('/home/fuxueping/sdb/Caffe_Project/face_recognition/test/result/diff_err_mobileface.txt','w')

    regmat = np.array(regfeat)
    capmat = np.array(capfeat)

    regmat_T = regmat.T
    SimilarityMatrix = np.dot(capmat, regmat_T)#dot()返回的是两个矩阵的乘积
    sameListscore = []
    diffListscore = []
    for i in range(len(capid)):
        for j in range(len(regid)):
            if capid[i] == regid[j]:
                sameListscore.append(SimilarityMatrix[i,j])
                if SimilarityMatrix[i, j] <= threshold_same:
                   f_same.writelines(capimg[i] + ' ' + regimg[j] + ' ' + str(SimilarityMatrix[i, j]) + '\n')

            else:
                diffListscore.append(SimilarityMatrix[i,j])
                if SimilarityMatrix[i, j] >= threshold_diff:
                    f_diff.writelines(capimg[i]+' '+regimg[j]+' '+str(SimilarityMatrix[i, j])+'\n')

    sameNum = len(sameListscore)
    diffNum = len(diffListscore)
    ratio = sameNum / diffNum
    print('\n')

    sigma = 0
    s1 = np.array(sameListscore)#np.array 存储单一数据类型的多维数组。
    s2 = np.array(diffListscore)
    for i in range(steps):
        sigma += step   # step = 0.01

        s1[s1 < sigma] = 0
        TPnum = np.count_nonzero(s1) #同一人高于sigma
        FNnum = sameNum - TPnum   #同一人低于sigma

        s2[s2 < sigma] = 0
        FPnum = np.count_nonzero(s2) #不同人高于sigma
        TNnum = diffNum - FPnum #不同人小于sigma

        if TPnum + FNnum != 0 and FPnum + TNnum != 0:
            fpr = FPnum *1.0 / (TNnum + FPnum)
            tpr = TPnum *1.0 / (TPnum + FNnum)
            acc = (TPnum + TNnum * ratio) / ((TPnum + FNnum) + (TNnum + FPnum) * ratio)
            re_fpr.append(fpr)
            re_tpr.append(tpr)
            re_acc.append(acc)
            min_fpr.append(abs(fpr-fpr_TH))
            reStr = "sigma= " + str(sigma) + " , fpr= " + str(fpr) + " , tpr= " + str(tpr) + " , acc= " + str(acc) + " , FP= " + str(FPnum) + " , TP= " + str(TPnum) + \
                    ', FP+FN='+ str(FPnum+sameNum-TPnum)
            print(reStr)
            f_re.writelines(reStr + '\n')
        else:
            print ("result file is wrong")
            exit(-1)

    final_001_fpr_index = min_fpr.index(min(min_fpr))
    final_001_fpr = re_fpr[final_001_fpr_index]
    final_sigma_001 = final_001_fpr_index * step + step
    final_001_tpr = re_tpr[final_001_fpr_index]
    # final_zero_fpr_index = re_fpr.index(0, 1)
    # final_zero_sigma = final_zero_fpr_index * step + step
    # final_zero_tpr = re_tpr[final_zero_fpr_index]

    reStr = "same pairs = " + str(sameNum) + " , diff pairs = " + str(diffNum) + " ,  ratio = " + str(ratio)
    print(reStr)
    f_re.writelines(reStr + '\n')
    reStr = 'The fpr that you care : ' + str(final_001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_001) + '   and the corresponding tpr : ' + str(final_001_tpr) + '\n' #+ \
            # 'The fpr that you care : ' + str(0) + '   and the corresponding sigma : ' + str(final_zero_sigma) + '   and the corresponding tpr : ' + str(final_zero_tpr)
    f_re.writelines(reStr + '\n')

    print(reStr)
    return s1, s2


def test_top1_identification(regfeat,capfeat,regid,capid,resultTxt):

    re_fpr, re_tpr, min_fpr = [], [], []
    f_re = open(resultTxt, 'w')

    regmat = np.array(regfeat)
    capmat = np.array(capfeat)

    regmat_T = regmat.T
    SimilarityMatrix = np.dot(capmat, regmat_T)#dot()返回的是两个数组的点积(dot product)

    top_id_List = []
    top_score_List = []
    top_im_List = []
    cap_id_List = capid
    rec_id_List = regid
    # rec_im_List = regimg
    # cap_im_List = capimg
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
    for i in range(steps):
        sigma += step  # step = 0.01
        FPNum, TPNum = 0, 0
        for i in range(len(capid)):
            if cap_id_List[i] != top_id_List[i] and top_score_List[i] > sigma:
                FPNum +=1
                # im_pair = (cap_im_List[i],top_im_List[i])
            elif cap_id_List[i] == top_id_List[i] and top_score_List[i]> sigma:
                TPNum +=1


        fpr = FPNum/total_num
        tpr = TPNum/in_num
        min_fpr.append(abs(fpr - fpr_TH))
        re_fpr.append(fpr)
        re_tpr.append(tpr)

        reStr = "sigma= " + str(sigma) + " , FPNum= " + str(FPNum)+ " , TPNum= " + str(TPNum)+ " , FPR= " + str(fpr) + " , TPR= " + str(tpr)
        print (reStr)
        f_re.writelines(reStr + '\n')

    final_001_fpr_index = min_fpr.index(min(min_fpr))
    final_001_fpr = re_fpr[final_001_fpr_index]
    final_001_tpr = re_tpr[final_001_fpr_index]
    final_sigma_001 = final_001_fpr_index * step + step
    print("all cap num:%d,cap in reg num:%d\n" % (total_num, in_num))
    reStr = 'The fpr that you care : ' + str(final_001_fpr) + '   and the corresponding sigma : ' + str(final_sigma_001) + '   and the corresponding tpr : ' + str(final_001_tpr) + '\n'
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
    print("S_max: ",S_max,"    S_min: ",S_min,"    ;     D_max: ",D_max,"    D_min: ",D_min)
    print("s_mean: ", s_mean, "    s_var: ", s_var, "    ;     d_mean: ", d_mean, "    d_var: ", d_var)
#    plt.hist(same_score, bins=100, facecolor="blue", edgecolor="black", alpha=0.7)
#    plt.xlabel("score")
#    plt.ylabel("number")
#    plt.title("same pairs seq score histogram")label='-same',
#    plt.show()
    plt.title("same/different pairs seq score histogram")
    plt.hist(same_score, bins=100, normed = 1, facecolor="red", edgecolor="black", label='same', alpha=0.7, hold = 1)
    plt.hist(diff_score, bins=100, normed = 1, facecolor="blue", edgecolor="black",label='different', alpha=0.7)
    # plt.hist(same_score, bins=100, facecolor="red", edgecolor="black", label='same', alpha=0.7)
    # plt.hist(diff_score, bins=100, facecolor="blue", edgecolor="black",label='different', alpha=0.7)
    plt.xlabel("score")
    plt.ylabel('pairs')
    plt.yticks([])
    plt.legend()
    plt.show()

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

    # net = initilizeNet_mobileFace()
    net = initilizeNet()
    resultPath =  '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/result/{}/sphere_res50.txt'.format(meth)

    # pairpath = '/home/nyy/DataSets/FPpairs_0.60'
    # file = 'diff_err.txt'

    if meth == 'identification':

            regbase ='/home/fuxueping/sdb/Caffe_Project/face_recognition/test/identification_{}_{}/reg'.format(imgSize[0],imgSize[1])
            capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/identification_{}_{}/cap'.format(imgSize[0],imgSize[1])

            # regbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/reg'
            # capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/cap'


            regimglist, regidlist = traversalDir_FirstDir(regbase)
            # regfeat = get_features_SEresnet50(net, regimglist, batch)
            regfeat = extractFeature_mobileFace(net, regimglist, batch)
            #
            capimglist, capidlist = traversalDir_FirstDir(capbase)
            # capfeat = get_features_SEresnet50(net, capimglist, batch)
            capfeat = extractFeature_mobileFace(net, capimglist, batch)

            test_top1_identification(regfeat, capfeat, regidlist, capidlist, resultPath)


    else:

            regbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/verification_{}_{}/TestID/ID'.format(imgSize[0],imgSize[1])
            capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/verification_{}_{}/TestID/face'.format(imgSize[0],imgSize[1])

            # regbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/reg'
            # capbase = '/home/fuxueping/sdb/Caffe_Project/face_recognition/test/kaoqin_112/cap'
            regimglist, regidlist = traversalDir_FirstDir(regbase)
            regfeat = extractFeature_mobileFace(net, regimglist, batch)
            # regfeat = get_features_SEresnet50(net, regimglist, batch)

            capimglist, capidlist = traversalDir_FirstDir(capbase)
            capfeat = extractFeature_mobileFace(net, capimglist, batch)
            # capfeat = get_features_SEresnet50(net, capimglist, batch)

            same_score, diff_score = test_id_verification(regfeat, capfeat, regidlist, capidlist, regimglist,
                              capimglist, resultPath)

            #Score_analyse(same_score, diff_score)
            # copy_pairlist(file,pairpath)


if __name__ == "__main__":
    method = 'identification'#识别1vsN
    # method = 'verification'  # 验证1vs1

    test(method)



"""
#获取单张图像的特征（caffe）
def caffe_get_feature(net, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125
    tempimg = np.zeros((1, 112, 112, 3))
    tempimg[0, :, :, :] = img
    tempimg = tempimg.transpose(0, 3, 1, 2)
    net.blobs['data'].data[...] = tempimg
    net.forward()
    features = copy.deepcopy(net.blobs['fc1'].data[...])
    feature = np.array(features[0])
    l2_norm = cv2.norm(feature, cv2.NORM_L2)
    return feature / l2_norm
"""

"""
    img_path = "/home/fuxueping/sdb/Caffe_Project/face_recognition/train/faces_ms1m_112x112/imgs/1/118.jpg"
    img1 = cv2.imread(img_path)

    plt.subplot(1, 3, 1)
    plt.title('cv2')
    plt.imshow(img1)


    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("1.jpg", img1)
    plt.subplot(1, 3, 2)
    plt.title('plt')
    plt.imshow(img)

    anc_img = Image.open(img_path)
    plt.subplot(1, 3, 3)
    plt.title('plt1')
    plt.imshow(anc_img)
    cv2.waitKey(1000)
    cv2.waitKey(1000)
"""