import argparse
import datetime
import math
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

import somatic_data_loader
from MyLogger import Logger
from nn_net import IndelNet, Net
from somatic_data_loader import Dataset, FastvcCallLoader


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

sys.stdout = Logger(filename = "./logs/mcall_somatic.out")
def ensumble_infer(inputs, nets, batchsize):
    predicted_sum = torch.zeros(inputs.shape[0])
    #for i in range(1, 10):
    for net in nets:
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_sum += predicted

    if use_cuda:
        predicted_sum = predicted_sum.cpu()
    predicted_sum = predicted_sum.numpy()
    positive_index = np.where(predicted_sum >= 15)
    return positive_index 


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
    out_dir = os.path.join(base_path, args.model_out)
    if not os.path.exists(out_dir):
        print("dir {} not exists!".format(out_dir))
        exit(-1)
    fastvc_result_path = args.in_data #[CHANGE]
    VarType = args.var_type #SNV or INDEL
    batch_size = args.batch_size
    nthreads = args.nthreads
    checkpoint = args.trained_model
    out_file = args.out
    #--------------------------------------------------------#
    
    reload_from_dupfile = False #load from file(True) or compute generate data again(Fasle)
    data_path = "./call_dataset.pkl"
    if args.re_exec:
        dataset = Dataset(reload_from_dupfile, False, args.re_exec, VarType, [region_file, fasta_file, bam_file], 
                                base_path, truth_path)
    else:
        dataset = Dataset(reload_from_dupfile, False, args.re_exec, VarType, [fastvc_result_path],
                                base_path, truth_path)
    if reload_from_dupfile:
        dataset.load(data_path)
    else:
        if os.path.exists(data_path):
            os.remove(data_path)
        #dataset.split(random_state = None)
        #dataset.store(data_path)
    #------------------------network setting---------------------#
    device = torch.device('cpu')
    n_feature = 0
    if VarType == "INDEL":
        n_feature = somatic_data_loader.SOM_INDEL_FEATURES 
        nets = list()
        for net_index in range(1, 17):
            model_name = checkpoint + "_round{}.pth".format(net_index)
            net = IndelNet(n_feature, [140, 160, 170, 100, 10] , 2)
            pretrained_dict = torch.load(model_name, map_location = device)
            pretrained_state_dict = pretrained_dict["state_dict"]
            net.load_state_dict(pretrained_state_dict)
            net.eval()
            nets.append(net)
    elif VarType == "SNV":
        n_feature = somatic_data_loader.SOM_SNV_FEATURES
        nets = list()
        for net_index in range(1, 16):
            model_name = checkpoint + "_round{}.pth".format(net_index)
            print("loading ", model_name)
            net = Net(n_feature, [140, 160, 170, 100, 10] , 2)
            #net = Net(n_feature, [80, 120, 140, 120, 200] , 2)
            pretrained_dict = torch.load(model_name, map_location = device)
            pretrained_state_dict = pretrained_dict["state_dict"]
            net.load_state_dict(pretrained_state_dict)
            net.eval()
            nets.append(net)
    else:
        print("illegal VarType: {} !!".format(VarType))
        exit(0)
    #------------------------------------------------------------#
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

        result_indexs.update(set(raw_indexs.numpy()[ensumble_infer(inputs, nets, batch_size)]))
        '''
        outputs = nets[0](inputs) #outputs is the prob of each class(P or N)
        # _, predicted = torch.max(outputs, 1)
        # _, predicted2 = torch.max(outputs2, 1)
        # _, predicted3 = torch.max(outputs3, 1)
        # predicted_sum = predicted + predicted2 + predicted3
        _, predicted_sum = torch.max(outputs, 1)
    
        if use_cuda:
           predicted = predicted.cpu() 
        predicted_sum = predicted_sum.numpy()
        positive_index = np.where(predicted_sum == 1)
        #print(len(positive_index[0]), positive_index[0])
        #print("raw index",set(raw_indexs.numpy()[positive_index]))
        result_indexs.update(set(raw_indexs.numpy()[positive_index]))
        '''
    fout = open(out_file, 'w')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "train your network")
    parser.add_argument('--re_exec', help = "", action="store_true")
    parser.add_argument('--region_file', help = "region file(.bed file)", type=str, required = False)
    parser.add_argument('--ref_file', help = "reference file", type=str, required = False)
    parser.add_argument('--bam_file', help = "input alignment file(.bam)", type=str, required = False)
    parser.add_argument('--workspace', help = "workspace", type=str, required = True)
    parser.add_argument('--in_data', help = "RabbitVar intermidiate file(with fisher test)", type=str, required = True)
    parser.add_argument('--truth_file', help = "truth file / the ground truth(.vcf)", type=str, required = True)
    parser.add_argument('--model_out', help = "the path you want to store your model", type=str, default="./models")
    parser.add_argument('--var_type', help = "var type you want to train(SNV/INDEL)", type=str, required = True)
    #parser.add_argument('--var_type', help = "var type you want to train(SNV/INDEL)", type=str, required = True)
    parser.add_argument('--batch_size', help = "batch size", type=int, default=128)
    parser.add_argument('--nthreads', help = "number of thread", type=int, default=20)
    parser.add_argument('--trained_model', help = "pretrained model", type=str, required = False)
    parser.add_argument('--out', help = "filtered result path ", type=str, required = True)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    call_somatic(args, use_cuda)
