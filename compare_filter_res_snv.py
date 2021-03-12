import sys
import os

#fastvc_file = "/home/haoz/deepfilter/deepfiltered_out.indel.txt" 
#fastvc_file = "/home/haoz/data/somatic/FD_10_21_data/train2.txt" 
#fastvc_file = "/home/haoz/deepfilter/workspace/test/germ_filtered_snv.txt"
fastvc_file = "/home/haoz/data/test.txt" 
#fastvc_file = "/home/haoz/data/chm1_chm13.txt" 
truth_file = "/home/haoz/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf"
#truth_file = "/home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf"
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
                    site = chrom + ":" + pos + ":" + ref.upper() + ":" + alt_i.upper()
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
        #if items[54] == 'SNV': #germline is item[35]
        if items[35] == 'SNV': #germline is item[35]
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

#print("before: total:{}\ttp:{}\tfp:{}".format(848068, 1722, 846346))#trainset 10-18
print("before: total:{}\ttp:{}\tfp:{}".format(2122678, 4419, 2118259))
print("after : total:{}\ttp:{}\tfp:{}".format(len(fastvc_snvs), fvc_cnt, len(fastvc_snvs) - fvc_cnt))
#test_total_truth = 129557
fvc_total_truth = 4419
print("truth indel num: {}, fastvc total output: {}\nfastvc find: {} recall: {}, prec: {}"\
    .format(len(truth_indels), len(fastvc_snvs), fvc_cnt, fvc_cnt / fvc_total_truth, fvc_cnt / len(fastvc_snvs)))
