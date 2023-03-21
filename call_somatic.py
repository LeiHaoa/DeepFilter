import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utils'))
from vcf_writer import *

import somatic_data_loader
from somatic_data_loader import Dataset, FastvcCallLoader
#from MyLogger import Logger
from nn_net import Net, IndelNet
import math
import argparse
import time

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


#sys.stdout = Logger(filename = "./logs/mcall_indel.out")

def call_somatic(args, use_cuda):
    #--------------------------------------------------------#
    #region_file = "/home/old_home/haoz/workspace/data/NA12878/ConfidentRegions.bed"
    #fasta_file = "/home/old_home/haoz/workspace/data/hg38/hg38.fa"
    #bam_file = "/home/old_home/haoz/workspace/data/NA12878/NA12878_S1.bam"
    #base_path = "/home/haoz/deepfilter/workspace"
    #models_dir = os.path.join(base_path, "models")
    #re_exec = False
    #strelka2_result_path = "/home/old_home/haoz/workspace/VCTools/strelka-2.9.10.centos6_x86_64/hg38run_40th/results/variants/variants.vcf"
    ##fastvc_result_path = "/home/haoz/data/lh_fisher.txt"
    ##truth_path =  "/home/haoz/data/full.37m.vcf"
    ##fastvc_result_path = "/home/haoz/data/out_fisher.vcf"
    #fastvc_result_path = "/home/haoz/data/somatic/FD_10_18_data/test.txt"
    ##fastvc_result_path = "/home/haoz/data/somatic/FDSynthetic.notloose.txt"
    ##fastvc_result_path = "/home/haoz/data/chm1_chm13.txt"
    #truth_path =  "/home/haoz/data/somatic/synthetic_indels.leftAlign.vcf"
    #checkpoint_w1_10 = os.path.join(models_dir, "checkpoint_INDEL_20-10-21-13-13-07_ecpch10.pth")
    #
    #output_path = "./deepfiltered_out.indel.txt"
    ##checkpoint = os.path.join(models_dir, "checkpoint_fastvc_20-09-21-01-04-02_ecpch93.pth")
    #--------------------------------------------------------#
    if args.re_exec:
        region_file = args.region_file
        fasta_file = args.ref_file
        bam_file = args.bam_file
    base_path = args.workspace
    truth_path =  args.truth_file
    #out_dir = os.path.join(base_path, args.model_out)
    #if not os.path.exists(out_dir):
    #    print("dir {} not exists!".format(out_dir))
    #    exit(-1)
    fastvc_result_path = args.in_data #[CHANGE]
    VarType = args.var_type #SNV or INDEL
    batch_size = args.batch_size
    nthreads = args.nthreads
    checkpoint = args.trained_model
    out_file = args.out
    #--------------------------------------------------------#
    loaddata_time_start = time.time() 
    reload_from_dupfile = False #load from file(True) or compute generate data again(Fasle)
    data_path = "./call_dataset.pkl"
    is_train = False
    if args.re_exec:
        dataset = Dataset(is_train, args.re_exec, VarType, [region_file, fasta_file, bam_file], 
                                base_path, truth_path)
    else:
        dataset = Dataset(is_train, args.re_exec, VarType, [fastvc_result_path],
                                base_path, truth_path)
        
    loaddata_time_end = time.time() 
    print("time of load and preprocessing data: {} s".format(loaddata_time_end - loaddata_time_start))
    #------------------------network setting---------------------#
    #n_feature = somatic_data_loader.SOM_INDEL_FEATURES if VarType == "INDEL" else somatic_data_loader.SOM_SNV_FEATURES
    ##net = Net(n_feature, [40, 60, 70, 60, 100] , 2)
    #net = Net(n_feature, [80, 120, 140, 120, 200] , 2)
    n_feature = 0
    if VarType == "INDEL":
        n_feature = somatic_data_loader.SOM_INDEL_FEATURES 
        net = IndelNet(n_feature, [140, 160, 170, 100, 10] , 2)
    elif VarType == "SNV":
        n_feature = somatic_data_loader.SOM_SNV_FEATURES
        #net = Net(n_feature, [80, 120, 140, 120, 200] , 2)
        net = Net(n_feature,  [140, 160, 170, 100, 10], 2)
    else:
        print("illegal VarType: {} !!".format(VarType))
        exit(0)
    #------------------------------------------------------------#
    device = torch.device('cpu')
    pretrained_dict = torch.load(checkpoint, map_location = device)
    model_tag = pretrained_dict["tag"]
    epoch_num = pretrained_dict["epoch"]
    pretrained_state_dict = pretrained_dict["state_dict"]
    
    net.load_state_dict(pretrained_state_dict)
    net.eval()
    
    epoch_loss = [] 
    runing_loss = 0.0
    use_cuda = False
    total = 0
    truth = 0
    false = 0
    
    test_dataset = FastvcCallLoader(dataset.inputs) 
    loader = torch.utils.data.DataLoader(test_dataset, 
                                batch_size = batch_size, 
                                shuffle = False,
                                num_workers = nthreads, 
                                pin_memory = True)
    #test_call(loader, net)
    result_indexs = set() 
    pred_collection = np.asarray([])
    print("total length: ", test_dataset.__len__())
    infer_start = time.time()
    summm = 0
    for i, data in enumerate(loader, 0):
        #inputs, labels, raw_indexs = data
        inputs = data
        #inputs, labels = Variable(inputs).float(), Variable(labels).long()
        inputs = Variable(inputs).float()
        total += len(inputs)
        if use_cuda:
            inputs = inputs.cuda()
    
        outputs = net(inputs) #outputs is the prob of each class(P or N)
        _, predicted = torch.max(outputs, 1)
    
        if use_cuda:
           predicted = predicted.cpu() 
        predicted = predicted.numpy()
        summm += sum(predicted)
        pred_collection = np.append(pred_collection, predicted)
        #positive_index = np.where(predicted == 1)
        #print(len(positive_index[0]), positive_index[0])
        #result_indexs.update(set(raw_indexs.numpy()[positive_index]))
    assert len(pred_collection) == len(dataset.df)
    print("test summm: ", summm)
    print("sumary of truth: ", sum(pred_collection))
    dataset.df['pred'] = pred_collection
    infer_end = time.time()
    print("inference over, time: {}s".format(infer_end - infer_start))
    vcf = dataset.df[dataset.df['pred'] == 1].sort_values(['Chr', 'Start'])
    SampleName = dataset.df['Sample'].iloc[0]
    with open(out_file, 'w') as f:
        write_header(f)
        tmp = "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", SampleName]) #TODO what if user not specified a sample name???
        tmp += '\n'
        f.write(tmp)
        for i, record in vcf.iterrows():
            f.write(format_record(record) + '\n')

    write_end = time.time()
    print("write over, time: {}s".format(write_end - infer_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "train your network")
    parser.add_argument('--re_exec', help = "", action="store_true")
    parser.add_argument('--region_file', help = "region file(.bed file)", type=str, required = False)
    parser.add_argument('--ref_file', help = "reference file", type=str, required = False)
    parser.add_argument('--bam_file', help = "input alignment file(.bam)", type=str, required = False)
    parser.add_argument('--workspace', help = "workspace", type=str, required = True)
    parser.add_argument('--in_data', help = "RabbitVar intermidiate file(with fisher test)", type=str, required = True)
    parser.add_argument('--truth_file', help = "truth file / the ground truth(.vcf)", type=str, required = False)
    parser.add_argument('--model_out', help = "the path you want to store your model", type=str, default="./models")
    parser.add_argument('--var_type', help = "var type you want to train(SNV/INDEL)", type=str, required = True)
    #parser.add_argument('--var_type', help = "var type you want to train(SNV/INDEL)", type=str, required = True)
    parser.add_argument('--batch_size', help = "batch size", type=int, default=128)
    parser.add_argument('--nthreads', help = "number of thread", type=int, default=8)
    parser.add_argument('--trained_model', help = "pretrained model", type=str, required = False)
    parser.add_argument('--out', help = "filtered result path ", type=str, required = True)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    call_somatic(args, use_cuda)
