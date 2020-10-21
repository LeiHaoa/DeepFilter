import sys
import os

#fastvc_file = "/home/haoz/deepfilter/deepfiltered_out.indel.txt" 
#fastvc_file = "/home/haoz/deepfilter/deepfiltered_out.indel.txt" 
fastvc_file = "/home/haoz/deepfilter/workspace/test/filtered_snv.txt"
#fastvc_file = "/home/haoz/data/out_fisher.txt" 
#fastvc_file = "/home/haoz/data/chm1_chm13.txt" 
#fastvc_file = "/home/haoz/data/wgs_loose_goodvar.txt" 
#truth_file = "/home/old_home/haoz/workspace/data/NA12878/vcfs/NA12878_S1.vcf"
truth_file = "/home/haoz/data/somatic/FD_10_18_data/synthetic_snvs.vcf"
#truth_file = "/home/haoz/data/full.37m.vcf"
#--- open truth vcf file ---#
truth_indels = dict()
with open(truth_file, 'r') as f:
    for var in f:
        items = var.split('\t')
        if(len(items) == 10 ):
            chrom, pos, id, ref, alt, _, filter = items[:7]         
            if len(chrom) < 6 :
                for alt_i in alt.split(","):
                    site = chrom + ":" + pos + ":" + ref + ":" + alt_i
                    truth_indels[site] = list([ref, alt_i])

#--- open fastvc txt file ---#
fastvc_snvs = dict()
with open(fastvc_file, 'r') as f:
    for var in f:
        if var[0] == '#':
            continue
        items = var.split('\t')
        chrom, pos, ref, alt = items[2], items[3], items[5], items[6]
        #chrom, pos, ref, alt = items[0], items[1], items[3], items[4]
        #if True: #len(ref) > 1 or len(alt) > 1 :
        if items[54] == 'SNV': #germline is item[35]
            for alt_i in alt.split(","):
                site = chrom + ":" + pos + ":" + ref + ":" + alt
                fastvc_snvs[site] = list([ref, alt])

print("------- SNV compare result: --------")
fvc_cnt = 0
sk_cnt = 0
differ_count = 0
for k, v in truth_indels.items():
    if k in fastvc_snvs:
        fvc_cnt += 1

print("before: total:{}\ttp:{}\tfp:{}".format(848086, 891, 87195))
print("after : total:{}\ttp:{}\tfp:{}".format(len(fastvc_snvs), fvc_cnt, len(fastvc_snvs) - fvc_cnt))
#test_total_truth = 129557
fvc_total_truth = 891
print("truth indel num: {}, fastvc total output: {}\nfastvc find: {} recall: {}, prec: {}"\
    .format(len(truth_indels), len(fastvc_snvs), fvc_cnt, fvc_cnt / fvc_total_truth, fvc_cnt / len(fastvc_snvs)))
