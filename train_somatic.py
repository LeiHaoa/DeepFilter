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

import somatic_data_loader as data_loader
from somatic_data_loader import Dataset, FVC_INDEL_FEATURES, FVC_SNV_FEATURES, FastvcTrainLoader
from MyLogger import Logger
from nn_net import Net
import math
import argparse

log_tag = "train_snv_drop5_loose_1_100" #[CHANGE]
sys.stdout = Logger(filename = "./logs/{}_{}.txt".format(log_tag, datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")))
def train_epoch(train_loader, net, optimizer):
    epoch_loss = [] 
    runing_loss = 0.0
    use_cuda = False
    #optimizer.zero_grad()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data #TODO #DONE
        inputs, labels = Variable(inputs).float(), Variable(labels).long()
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs) #outputs is the prob of each class(P or N)
        loss_ = loss_func(outputs, labels)
        loss_.backward()
        #update wight
        optimizer.step()

        epoch_loss.append(loss_.item())
        runing_loss += loss_.item()

        if i % 10000 == 9999:
            print("[%5d] loss: %.3f" % (i + 1, runing_loss / 10000))
            #np.set_printoptions(precision = 3, threshold=10000)
            #print("[inputs info]: \n {}".format(inputs.data.cpu().numpy()))
            #print("[pred info]:\n {}".format(outputs.data.cpu().numpy()))
            #print("[label info]:\n {}".format(labels.data.cpu().numpy()))
            #print('[net info]:')
            #for layer in net.modules():
            #    if isinstance(layer, nn.Linear):
            #        print(layer.weight)
            runing_loss = 0.0

    return epoch_loss

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
def test_epoch(test_loader, net, optimizer):
    epoch_loss = [] 
    runing_loss = 0.0
    #optimizer.zero_grad()
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
        #print(predicted.is_cuda, labels.is_cuda)
        t00, t01, t10, t11 = print_cmp2x2(predicted, labels)
        g00 += t00
        g01 += t01
        g10 += t10
        g11 += t11
        compare_labels = (predicted == labels)
        false_preds = np.where(compare_labels.data.cpu().numpy() == 0)[0]
        false += len(false_preds)
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

    return haoz_feature

def train_somatic(args, use_cuda = False):
    #--------------------------------------------------------#
    #region_file = "/home/old_home/haoz/workspace/data/NA12878/ConfidentRegions.bed"
    #fasta_file = "/home/old_home/haoz/workspace/data/hg38/hg38.fa"
    #bam_file = "/home/old_home/haoz/workspace/data/NA12878/NA12878_S1.bam"
    #truth_path =  "/home/haoz/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf"
    #base_path = "/home/haoz/deepfilter/workspace"
    if args.re_exec:
        region_file = args.region_file
        fasta_file = args.ref_file
        bam_file = args.bam_file
    base_path = args.workspace
    truth_path =  args.truth_file
    out_dir = os.path.join(base_path, args.model_out)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    #fastvc_result_path = "/home/haoz/data/out_fisher.vcf"
    #fastvc_result_path = "/home/haoz/data/wgs_loose_goodvar.txt"
    fastvc_result_path = args.train_data #[CHANGE]
    VarType = args.var_type #SNV or INDEL
    batch_size = args.batch_size
    nthreads = args.nthreads
    #--------------------------------------------------------#
    
    reload_from_dupfile = False #load from file(True) or compute generate data again(Fasle)
    data_path = "./dataset_{}.pkl".format(VarType)
    if args.re_exec:
        dataset = Dataset(reload_from_dupfile, args.re_exec, VarType, [region_file, fasta_file, bam_file], 
                                base_path, truth_path)
    else:
        dataset = Dataset(reload_from_dupfile, args.re_exec, VarType, [fastvc_result_path],
                                base_path, truth_path)
    if reload_from_dupfile:
        dataset.load(data_path)
    else:
        if os.path.exists(data_path):
            os.remove(data_path)
        dataset.split(random_state = None)
        dataset.store(data_path)
    #------------------------network setting---------------------#
    max_epoch = 10 
    save_freq = 10 # save every xx save_freq
    n_feature = data_loader.SOM_INDEL_FEATURES if VarType == "INDEL" else data_loader.SOM_SNV_FEATURES
    print("[info] n_feature: ", n_feature)
    net = Net(n_feature, [140, 160, 170, 100, 10] , 2)
    if use_cuda:
        net.cuda()
    #------------------------------------------------------------#
    if args.pretrained_model != "": #if use pretrained model, load it
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        pretrained_dict = torch.load(args.pretrained_model, map_location = device)
        #model_tag = pretrained_dict["tag"]
        #epoch_num = pretrained_dict["epoch"]
        pretrained_state_dict = pretrained_dict["state_dict"]
        net.load_state_dict(pretrained_state_dict)
        net.eval()
    else: #else init the net
        net.initialize_weights()

    n_epoch = 0
    #optimizer
    #optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
    #optimizer = optim.Adam(net.parameters(), lr = 0.01)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=0.1, rho=0.96, eps=1e-05, weight_decay=0)
    weight = torch.Tensor([1, 10]) #[CHANGE]
    if(use_cuda):
        weight = weight.cuda()
    #loss_func = torch.nn.MSELoss()
    #loss_func = torch.nn.BCELoss()
    loss_func = torch.nn.CrossEntropyLoss(weight = weight)
    #------------------------------------------------------------#
    
    max_haoz_feature = 0
    for epoch in range(max_epoch):
        print("epoch", epoch, " processing....")
        if (not reload_from_dupfile) and (epoch != 0):
            dataset.split(random_state = None)
        test_dataset = FastvcTrainLoader(dataset.data['test'])
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                    batch_size = batch_size, shuffle = True,
                                    num_workers = nthreads, 
                                    pin_memory = True)
        train_dataset = FastvcTrainLoader(dataset.data['train'])
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                    batch_size = batch_size, shuffle = True,
                                    num_workers = nthreads, 
                                    pin_memory = True)
    
        n_epoch += 1
        #epoch_loss = train_epoch(train_loader, net, optimizer)
        #------------------------train epoch-----------------
        epoch_loss = [] 
        runing_loss = 0.0
        #optimizer.zero_grad()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data 
            inputs, labels = Variable(inputs).float(), Variable(labels).long()
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs) #outputs is the prob of each class(P or N)
            loss_ = loss_func(outputs, labels)
            loss_.backward()
            #update wight
            optimizer.step()
    
            epoch_loss.append(loss_.item())
            runing_loss += loss_.item()
    
            if i % 1000 == 999:
                print("[%5d] loss: %.3f" % (i + 1, runing_loss / 999))
                runing_loss = 0.0
    
        print("mean loss of epoch %d is: %f" % (epoch, sum(epoch_loss) / len(epoch_loss)))
        epoch_feature = test_epoch(test_loader, net, optimizer)
        if n_epoch == - 1: # (save_freq - 1):
        #if epoch_feature > max_haoz_feature:
            max_haoz_feature = epoch_feature
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            tag = "fastvc_{}".format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
            torch.save({
                "state_dict": net.state_dict(),
                "tag": tag,
                "epoch": n_epoch,
                }, '{}/checkpoint_{}_ecpch{}.pth'.format(out_dir, tag, n_epoch))
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    tag = "fastvc_{}".format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
    torch.save({
        "state_dict": net.state_dict(),
        "tag": tag,
        "epoch": n_epoch,
        }, '{}/checkpoint_{}_ecpch{}.pth'.format(out_dir, tag, n_epoch))
    print("training done!")
    print("final model:", '{}/models/checkpoint_{}_ecpch{}.pth'.format(out_dir, tag, n_epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "train your network")
    parser.add_argument('--re_exec', help = "", action="store_true")
    parser.add_argument('--region_file', help = "region file(.bed file)", type=str, required = False)
    parser.add_argument('--ref_file', help = "reference file", type=str, required = False)
    parser.add_argument('--bam_file', help = "input alignment file(.bam)", type=str, required = False)
    parser.add_argument('--workspace', help = "workspace", type=str, required = True)
    parser.add_argument('--train_data', help = "RabbitVar intermidiate file(with fisher test)", type=str, required = True)
    parser.add_argument('--truth_file', help = "truth file / the ground truth(.vcf)", type=str, required = True)
    parser.add_argument('--model_out', help = "the path you want to store your model", type=str, default="./models")
    parser.add_argument('--var_type', help = "var type you want to train(SNV/INDEL)", type=str, required = True)
    #parser.add_argument('--var_type', help = "var type you want to train(SNV/INDEL)", type=str, required = True)
    parser.add_argument('--batch_size', help = "batch size", type=int, default=128)
    parser.add_argument('--nthreads', help = "number of thread", type=int, default=20)
    parser.add_argument('--pretrained_model', help = "pretrained model", type=str, required = False)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    train_somatic(args, use_cuda)