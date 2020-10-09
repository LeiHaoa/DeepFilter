import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import transforms

import data_loader
from data_loader import Dataset, FastvcCallLoader
from MyLogger import Logger
from nn_net import Net
import math

def print_cmp2x2(p, l):
    p = p.cpu().numpy()
    l = l.cpu().numpy()
    g00, g01, g10, g11 = 0, 0, 0, 0
    assert(len(p) == len(l))
    for i in range(len(p)):
        if p[i] == 0:
            if l[i] == 0:
                g00 += 1
            else:
                g01 += 1
        else:
            if l[i] == 0:
                g10 += 1
            else:
                g11 += 1
    return g00, g01, g10, g11

def test_call(test_loader, net):
    runing_loss = 0.0
    g00, g01, g10, g11 = 0, 0, 0, 0
    total = 0
    truth = 0
    false = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data #TODO #DONE
        inputs, labels = Variable(inputs).float(), Variable(labels).long()
        total += len(inputs)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = net(inputs) #outputs is the prob of each class(P or N)
        _, predicted = torch.max(outputs, 1)
        t00, t01, t10, t11 = print_cmp2x2(predicted, labels)
        g00 += t00
        g01 += t01
        g10 += t10
        g11 += t11
        compare_labels = (predicted == labels)
        false_preds = np.where(compare_labels.data.cpu().numpy() == 0)[0]
        false += len(false_preds)
        if i % 10000 == 0:
            print("P|L\t0\t1\n0\t{}\t{}\n1\t{}\t{}".format(g00, g01, g10, g11))
    print("test info:\n [total]:{}, [false]:{}, [truth]:{}, error rate:{}".format(total, false, total - false, false/total) )
    print("P\\L\t0\t1\n0\t{}\t{}\n1\t{}\t{}".format(g00, g01, g10, g11))
    TNR =  g01 / (g01 + g10)
    print("[01/(01+10)] TNR:\t ", TNR)
    haoz_feature = -1 * math.log((TNR + 0.000001) * (false/total))
    print("[00/(10+00)] filtered False rate:\t {}".format(g00 /(g10 + g00)))
    print("[01/(01+11)] filtered Truth rate:\t {}".format(g01 / (g01 + g11)))
    print("[11/(10+11)] Precision:\t {}".format(g10 / (g10 + g11)))
    print("[11/(01+11)] Recall:\t {}".format(g11 / (g01 + g11)))
    print("haoz feature(bigger better):\t ", haoz_feature)


sys.stdout = Logger(filename = "./logs/mcall_snv.out")
#--------------------------------------------------------#
region_file = "/home/old_home/haoz/workspace/data/NA12878/ConfidentRegions.bed"
fasta_file = "/home/old_home/haoz/workspace/data/hg38/hg38.fa"
bam_file = "/home/old_home/haoz/workspace/data/NA12878/NA12878_S1.bam"
base_path = "/home/haoz/python/workspace"
models_dir = os.path.join(base_path, "out/models")
re_exec = False
strelka2_result_path = "/home/old_home/haoz/workspace/VCTools/strelka-2.9.10.centos6_x86_64/hg38run_40th/results/variants/variants.vcf"
#fastvc_result_path = "/home/haoz/data/lh_fisher.txt"
#truth_path =  "/home/haoz/data/full.37m.vcf"
#fastvc_result_path = "/home/haoz/data/out_fisher.vcf"
#fastvc_result_path = "/home/haoz/data/test.txt"
fastvc_result_path = "/home/haoz/data/wgs_loose_goodvar.txt"
#fastvc_result_path = "/home/haoz/data/chm1_chm13.txt"
truth_path =  "/home/haoz/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf"
checkpoint_w1_1 = os.path.join(models_dir, "checkpoint_fastvc_20-10-06-16-53-05_ecpch10.pth")
checkpoint_w1_10 = os.path.join(models_dir, "checkpoint_fastvc_20-10-06-17-47-16_ecpch10.pth")
checkpoint_w1_100 = os.path.join(models_dir, "checkpoint_fastvc_20-10-06-20-26-58_ecpch10.pth")

output_path = "./deepfiltered_out.snv.txt"
#checkpoint = os.path.join(models_dir, "checkpoint_fastvc_20-09-21-01-04-02_ecpch93.pth")
#--------------------------------------------------------#

reload_from_dupfile = False #load from file(True) or compute generate data again(Fasle)
data_path = "./call_dataset.pkl"
VarType = 'SNV' #'SNV' or 'INDEL'
if re_exec:
    dataset = Dataset(reload_from_dupfile, re_exec, VarType, [region_file, fasta_file, bam_file], 
                            base_path, truth_path)
else:
    dataset = Dataset(reload_from_dupfile, re_exec, VarType, [fastvc_result_path, strelka2_result_path],
                            base_path, truth_path)
if reload_from_dupfile:
    dataset.load(data_path)
else:
    if os.path.exists(data_path):
        os.remove(data_path)
    #dataset.split(random_state = None)
    #dataset.store(data_path)
#------------------------network setting---------------------#
n_feature = data_loader.FVC_INDEL_FEATURES if VarType == "INDEL" else data_loader.FVC_SNV_FEATURES
net = Net(n_feature, [140, 160, 170, 100, 10] , 2)
net2 = Net(n_feature, [140, 160, 170, 100, 10] , 2)
net3 = Net(n_feature, [140, 160, 170, 100, 10] , 2)
use_cuda = True
batch_size = 128 
nthreads = 20
#------------------------------------------------------------#
device = torch.device('cpu')
#--- 1:1 network ---#
pretrained_dict = torch.load(checkpoint_w1_1, map_location = device)
model_tag = pretrained_dict["tag"]
epoch_num = pretrained_dict["epoch"]
pretrained_state_dict = pretrained_dict["state_dict"]
#--- 1:10 network ---#
pretrained_dict2 = torch.load(checkpoint_w1_10, map_location = device)
pretrained_state_dict2 = pretrained_dict2["state_dict"]
#--- 1:100 network ---#
pretrained_dict3 = torch.load(checkpoint_w1_100, map_location = device)
pretrained_state_dict3 = pretrained_dict3["state_dict"]

net.load_state_dict(pretrained_state_dict)
net.eval()
net2.load_state_dict(pretrained_state_dict2)
net2.eval()
net3.load_state_dict(pretrained_state_dict3)
net3.eval()

epoch_loss = [] 
runing_loss = 0.0
use_cuda = False
total = 0
truth = 0
false = 0

test_dataset = FastvcCallLoader([dataset.inputs, dataset.labels, dataset.raw_indexs]) 
loader = torch.utils.data.DataLoader(test_dataset, 
                            batch_size = batch_size, 
                            shuffle = False,
                            num_workers = nthreads, 
                            pin_memory = True)
#test_call(loader, net)
result_indexs = set() 
print("total length: ", test_dataset.__len__())
for i, data in enumerate(loader, 0):
    inputs, labels, raw_indexs = data #TODO #DONE
    inputs, labels = Variable(inputs).float(), Variable(labels).long()
    total += len(inputs)
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()

    outputs = net(inputs) #outputs is the prob of each class(P or N)
    outputs2 = net2(inputs) #outputs is the prob of each class(P or N)
    outputs3 = net3(inputs) #outputs is the prob of each class(P or N)
    outputs_sum = outputs + outputs2 + outputs3
    # _, predicted = torch.max(outputs, 1)
    # _, predicted2 = torch.max(outputs2, 1)
    # _, predicted3 = torch.max(outputs3, 1)
    # predicted_sum = predicted + predicted2 + predicted3
    _, predicted_sum = torch.max(outputs_sum, 1)

    if use_cuda:
       predicted = predicted.cpu() 
    predicted_sum = predicted_sum.numpy()
    positive_index = np.where(predicted_sum >= 0.1)
    #print(len(positive_index[0]), positive_index[0])
    #print("raw index",set(raw_indexs.numpy()[positive_index]))
    result_indexs.update(set(raw_indexs.numpy()[positive_index]))
fout = open(output_path, 'w')
rindex = 0
with open(fastvc_result_path, 'r') as f:
    for record in f:
        if rindex in result_indexs:
            fout.write(record)
        rindex += 1
fout.close()

'''
tmp_file = open("./tmp.out", 'w')
for i, data in enumerate(loader, 0):
    inputs, labels = data #TODO #DONE
    inputs = Variable(inputs).float()
    total += len(inputs)
    if use_cuda:
        inputs = inputs.cuda()

    outputs = net(inputs) #outputs is the prob of each class(P or N)
    _, predicted = torch.max(outputs, 1)
    if use_cuda:
        predicted = predicted.cpu()
    predicted = predicted.data.numpy()

tmp_file.close()
'''
