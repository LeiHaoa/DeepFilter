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
#from torchvision import transforms

import somatic_data_loader
from somatic_data_loader import Dataset, FastvcCallLoader
from MyLogger import Logger
from nn_net import Net, IndelNet
import math
import argparse
import time

def write_header(fout):
    header = """\
##fileformat=VCFv4.3
##INFO=<ID=SAMPLE,Number=1,Type=String,Description="Sample name (with whitespace translated to underscores)">
##INFO=<ID=TYPE,Number=1,Type=String,Description="Variant Type: SNV Insertion Deletion Complex">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=END,Number=1,Type=Integer,Description="Chr End Position">
##INFO=<ID=VD,Number=1,Type=Integer,Description="Variant Depth">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##INFO=<ID=BIAS,Number=1,Type=String,Description="Strand Bias Info">
##INFO=<ID=REFBIAS,Number=1,Type=String,Description="Reference depth by strand">
##INFO=<ID=VARBIAS,Number=1,Type=String,Description="Variant depth by strand">
##INFO=<ID=PMEAN,Number=1,Type=Float,Description="Mean position in reads">
##INFO=<ID=PSTD,Number=1,Type=Float,Description="Position STD in reads">
##INFO=<ID=QUAL,Number=1,Type=Float,Description="Mean quality score in reads">
##INFO=<ID=QSTD,Number=1,Type=Float,Description="Quality score STD in reads">
##INFO=<ID=SBF,Number=1,Type=Float,Description="Strand Bias Fisher p-value">
##INFO=<ID=ODDRATIO,Number=1,Type=Float,Description="Strand Bias Odds ratio">
##INFO=<ID=MQ,Number=1,Type=Float,Description="Mean Mapping Quality">
##INFO=<ID=SN,Number=1,Type=Float,Description="Signal to noise">
##INFO=<ID=HIAF,Number=1,Type=Float,Description="Allele frequency using only high quality bases">
##INFO=<ID=ADJAF,Number=1,Type=Float,Description="Adjusted AF for indels due to local realignment">
##INFO=<ID=SHIFT3,Number=1,Type=Integer,Description="No. of bases to be shifted to 3 prime for deletions due to alternative alignment">
##INFO=<ID=MSI,Number=1,Type=Float,Description="MicroSatellite. > 1 indicates MSI">
##INFO=<ID=MSILEN,Number=1,Type=Float,Description="MicroSatellite unit length in bp">
##INFO=<ID=NM,Number=1,Type=Float,Description="Mean mismatches in reads">
##INFO=<ID=LSEQ,Number=1,Type=String,Description="5' flanking seq">
##INFO=<ID=RSEQ,Number=1,Type=String,Description="3' flanking seq">
##INFO=<ID=GDAMP,Number=1,Type=Integer,Description="No. of amplicons supporting variant">
##INFO=<ID=TLAMP,Number=1,Type=Integer,Description="Total of amplicons covering variant">
##INFO=<ID=NCAMP,Number=1,Type=Integer,Description="No. of amplicons don't work">
##INFO=<ID=AMPFLAG,Number=1,Type=Integer,Description="Top variant in amplicons don't match">
##INFO=<ID=HICNT,Number=1,Type=Integer,Description="High quality variant reads">
##INFO=<ID=HICOV,Number=1,Type=Integer,Description="High quality total reads">
##INFO=<ID=SPLITREAD,Number=1,Type=Integer,Description="No. of split reads supporting SV">
##INFO=<ID=SPANPAIR,Number=1,Type=Integer,Description="No. of pairs supporting SV">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type: INV DUP DEL INS FUS">
##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="The length of SV in bp">
##INFO=<ID=DUPRATE,Number=1,Type=Float,Description="Duplication rate in fraction">
##FILTER=<ID=q{qmean},Description="Mean Base Quality Below $qmean">
##FILTER=<ID=Q{Qmean},Description="Mean Mapping Quality Below $Qmean">
##FILTER=<ID=p{Pmean},Description="Mean Position in Reads Less than $Pmean">
##FILTER=<ID=SN{SN},Description="Signal to Noise Less than $SN">
##FILTER=<ID=Bias,Description="Strand Bias">
##FILTER=<ID=pSTD,Description="Position in Reads has STD of 0">
##FILTER=<ID=d{TotalDepth},Description="Total Depth < $TotalDepth">
##FILTER=<ID=v{VarDepth},Description="Var Depth < $VarDepth">
##FILTER=<ID=f{Freq},Description="Allele frequency < $Freq">
##FILTER=<ID=MSI{opt_I},Description="Variant in MSI region with $opt_I non-monomer MSI or 13 monomer MSI">
##FILTER=<ID=NM{opt_m},Description="Mean mismatches in reads >= $opt_m, thus likely false positive">
##FILTER=<ID=InGap,Description="The variant is in the deletion gap, thus likely false positive">
##FILTER=<ID=InIns,Description="The variant is adjacent to an insertion variant">
##FILTER=<ID=Cluster${opt_c}bp,Description="Two variants are within $opt_c bp">
##FILTER=<ID=LongMSI,Description="The somatic variant is flanked by long A/T (>=14)">
##FILTER=<ID=AMPBIAS,Description="Indicate the variant has amplicon bias.">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=VD,Number=1,Type=Integer,Description="Variant Depth">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">
##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=RD,Number=2,Type=Integer,Description="Reference forward, reverse reads">
##FORMAT=<ID=ALD,Number=2,Type=Integer,Description="Variant forward, reverse reads">
""".format(xxx)
    fout.write(header)

def format_record(record):
    [sample, gene, chrt, start, end, ref, alt, dp1, vd1, rfwd1, rrev1, vfwd1, \
    vrev1, gt1, af1, bias1, pmean1, pstd1, qual1, qstd1, mapq1, sn1, hiaf1, \
    adjaf1, nm1, sbf1, oddratio1, dp2, vd2, rfwd2, rrev2, vfwd2, vrev2, \
    gt2, af2, bias2, pmean2, pstd2, qual2, qstd2, mapq2, sn2, hiaf2, \
    adjaf2, nm2, sbf2, oddratio2, shift3, msi, msilen, lseq, rseq, seg, \
    status, vtype, sv1, duprate1, sv2, duprate2, pvalue, oddratio]  = record.split("\t")
    rd1 = str(int(rfwd1) + int(rrev1))
    rd2 = str(int(rfwd2) + int(rrev2))
    if vtype == "" : 
        vtype = "REF"
    '''
    还有一些操作没做
        my $gt = (1-$af1 < $GTFREQ) ? "1/1" : ($af1 >= 0.5 ? "1/0" : ($af1 >= $FREQ ? "0/1" : "0/0"));
	    my $gtm = (1-$af2 < $GTFREQ) ? "1/1" : ($af2 >= 0.5 ? "1/0" : ($af2 >= $FREQ ? "0/1" : "0/0"));
        $bias1 =~ s/;/,/;
        $bias2 =~ s/;/,/;
        $bias1 = "0,0" if ($bias1 eq '0');
        $bias2 = "0,0" if ($bias2 eq '0');
        $mapq1 = sprintf '%.0f', $mapq1;
        $mapq2 = sprintf '%.0f', $mapq2;
	    my $qual = $vd1 > $vd2 ? int(log($vd1)/log(2) * $qual1) : int(log($vd2)/log(2) * $qual2);
    '''
    gt  = "1/1" if (1-af1 < GTFREQ) else ("1/0" if af1 >= 0.5 else ("0/1" if af1 >= FREQ else "0/0"))
    gtm = "1/1" if (1-af2 < GTFREQ) else ("1/0" if af2 >= 0.5 else ("0/1" if af2 >= FREQ else "0/0"))
    bias1 = bias1.replace(';',',')
    bias2 = bias2.replace(';',',')
    if bias1 == "0": bias1 = "0,0" 
    if bias2 == "0": bias2 = "0,0" 
    mapq1 = f'{mapq1:.0}'
    mapq2 = f'{mapq2:.0}'
    qual = int(math.log(vd1)/math.log(2) * qual1) if vd1 > vd2  else int(math.log(vd2)/math.log(2) * qual2)

    pinfo1 = "\t".join(chrt, start, ".", ref, alt, str(qual))
    filters = "PASS"

    pinfo2_1 = "STATUS={};SAMPLE={};TYPE={};DP={};VD={};AF={};SHIFT3={};MSI={};MSILEN={};SSF={};SOR={};LSEQ={};RSEQ={}".format(status, sample_nowhitespace, vtype, dp1, af1, shift3, msi, msilen, pvalue, oddratio, lseq, rseq)
    pinfo2_2 = ":".join(gt, dp1, vd1, vfwd1 + "," + vrev1, rfwd1+","+rrev1, rd1+","+vd1, af1, bias1, pmean1, pstd1, qual1, qstd1, sbf1, oddratio1, mapq1, sn1, hiaf1, adjaf1, nm1) 
    pinfo2_3 = ":".join(gtm, dp2, vd2, vfwd2 + "," + vrev2, rfwd2+","+rrev2, rd2+","+vd2, af2, bias2, pmean2, pstd2, qual2, qstd2, sbf2, oddratio2, mapq2, sn2, hiaf2, adjaf2, nm2)
    pinfo2 = "\t".join(pinfo2_1, "GT:DP:VD:ALD:RD:AD:AF:BIAS:PMEAN:PSTD:QUAL:QSTD:SBF:ODDRATIO:MQ:SN:HIAF:ADJAF:NM", pinfo2_2, pinfo2_3)
    return "\t".join(pinfo1, filters, pinfo2)


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


sys.stdout = Logger(filename = "./logs/mcall_indel.out")

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
        #dataset.split(random_state = None)
        #dataset.store(data_path)
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
    #--- 1:10 network ---#
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
    
    test_dataset = FastvcCallLoader([dataset.inputs, dataset.labels, dataset.raw_indexs]) 
    loader = torch.utils.data.DataLoader(test_dataset, 
                                batch_size = batch_size, 
                                shuffle = False,
                                num_workers = nthreads, 
                                pin_memory = True)
    #test_call(loader, net)
    result_indexs = set() 
    print("total length: ", test_dataset.__len__())
    infer_start = time.time()
    for i, data in enumerate(loader, 0):
        inputs, labels, raw_indexs = data #TODO #DONE
        inputs, labels = Variable(inputs).float(), Variable(labels).long()
        total += len(inputs)
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
    
        outputs = net(inputs) #outputs is the prob of each class(P or N)
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
    infer_end = time.time()
    print("inference over, time: {}s".format(infer_end - infer_start))
    #print("---->", result_indexs)
    write_start = time.time()
    fout = open(out_file, 'w')
    rindex = 0
    '''
    write_header(fout)
    tmp = "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "$sample"])
    tmp += '\n'
    fout.write(tmp)
    '''
    with open(fastvc_result_path, 'r') as f:
        for record in f:
            if rindex in result_indexs:
                #fout.write( format_record(record) )
                fout.write(record)
            rindex += 1
    fout.close()
    write_end = time.time()
    print("write over, time: {}s".format(write_end - write_start))


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
