import sys
import os

fastvc_file = "/home/haoz/deepfilter/workspace/test/filtered_snv.txt" 
truth_file = "/home/haoz/data/somatic/FD_10_18_data/synthetic_indels.leftAlign.vcf"
sk_file    = "/home/haoz/data/somatic/somatic.indels.vcf"
cmp_sk2 = False
cmp_fvc = True
#--- open truth vcf file ---#
truth_indels = dict()
with open(truth_file, 'r') as f:
    for var in f:
        items = var.split('\t')
        if(len(items) == 10 ):
            chrom, pos, id, ref, alt, _, filter = items[:7]         
            #if len(chrom) < 6 and filter == "PASS" and (len(ref) > 1 or len(alt) > 1) :
            if len(chrom) < 6 and (len(ref) > 1 or len(alt) > 1) :
                for alt_i in alt.split(","):
                    site = chrom + ":" + pos + ":" + ref + ":" + alt_i
                    truth_indels[site] = list([ref, alt_i])

if cmp_sk2:
    strelka_indels = dict()
    with open(sk_file, 'r') as f:
        for var in f:
            if var[0] == '#':
                continue
            items = var.split('\t')
            chrom, pos, id, ref, alt, _, filter = items[:7]         
            if len(chrom) < 6: #and filter ==  "PASS":
                for alt_i in alt.split(","):
                    site = chrom + ":" + pos + ":" + ref + ":" + alt_i
                    strelka_indels[site] = list([ref, alt_i])
if cmp_fvc:
    #--- open fastvc vcf file ---#
    fastvc_indels = dict()
    with open(fastvc_file, 'r') as f:
        for var in f:
            if var[0] == '#':
                continue
            items = var.split('\t')
            chrom, pos, ref, alt = items[2], items[3], items[5], items[6]
            #chrom, pos, ref, alt = items[0], items[1], items[3], items[4]
            if len(ref) > 1 or len(alt) > 1 :
                for alt_i in alt.split(","):
                    site = chrom + ":" + pos + ":" + ref + ":" + alt
                    fastvc_indels[site] = list([ref, alt])
    '''
    fastvc_indels = dict()
    with open(fastvc_file, 'r') as f:
        for var in f:
            if var[0] == '#':
                continue
            items = var.split('\t')
            chrom, pos, id, ref, alt, _, filter = items[:7]         
            #chrom, pos, ref, alt = items[0], items[1], items[3], items[4]
            if len(ref) > 1 or len(alt) > 1 :
                site = chrom + ":" + pos + ":" + ref + ":" + alt
                fastvc_indels[site] = list([ref, alt])
    '''


print("------- Indels compare result: --------")
fvc_cnt = 0
sk_cnt = 0
differ_count = 0
for k, v in truth_indels.items():
    if cmp_fvc and (k in fastvc_indels):
        fvc_cnt += 1
    if cmp_sk2 and (k in strelka_indels):
        sk_cnt += 1

test_total_truth = 129557
fvc_total_truth = 623 
if cmp_fvc:
    print("fastvc truth snv num: {}, fastvc total output: {}\nfastvc find: {} , filtered: {}, recall: {}, prec: {}"\
    .format(len(truth_indels), len(fastvc_indels), fvc_cnt, fvc_cnt / fvc_total_truth, fvc_cnt / len(fastvc_indels)))
if cmp_sk2:
    print("strelka truth snv num: {}, strelka total output: {}\nstrelka find: {} recall: {}, prec: {}"\
    .format(len(truth_indels), len(strelka_indels), sk_cnt, sk_cnt / fvc_total_truth, sk_cnt / len(strelka_indels)))
