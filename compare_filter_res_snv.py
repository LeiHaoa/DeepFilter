import sys
import os

#fastvc_file = "/home/haoz/data/somatic/FD1021_tumoronly/FDSynthetic1021_tumoronly.txt" 
fastvc_file = "/home/haoz/deepfilter/workspace/result/filtered_snv.txt"

#fastvc_file = "/home/haoz/data/test.txt" 
#truth_file = "/home/haoz/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf"
truth_file = "/home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf"
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
        if items[35] == 'SNV': #germline is item[35] somatic is item[54]
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

org_total = 6875866
org_tp    = 17425 
org_fp    = 6858441
#print("before: total:{}\ttp:{}\tfp:{}".format(848068, 1722, 846346))#trainset 10-18
print("before: total:{}\ttp:{}\tfp:{}, recall:{}, precision:{}".format(org_total, org_tp, org_fp, org_tp/len(truth_indels), org_tp/org_total))
print("after : total:{}\ttp:{}\tfp:{}".format(len(fastvc_snvs), fvc_cnt, len(fastvc_snvs) - fvc_cnt))
filter_prec = (org_fp - (len(fastvc_snvs)-fvc_cnt) + fvc_cnt) / org_total
print("filter Accuracy:", filter_prec)
total_truth = len(truth_indels)
print("truth snv num: {}, fastvc total output: {}\nfastvc find: {} filter_recall: {}, prec: {}, total_recall: {}"\
    .format(len(truth_indels), len(fastvc_snvs), fvc_cnt, fvc_cnt / org_tp, fvc_cnt / len(fastvc_snvs), fvc_cnt / total_truth))
