import sys
import os

sk2_file = "/home/old_home/haoz/workspace/VCTools/strelka-2.9.10.centos6_x86_64/hg38run_40th/results/variants/variants.vcf"
fastvc_file = "/home/haoz/python/deepfiltered_out.indel.txt" 
#fastvc_file = "/home/haoz/data/test.txt" 
#fastvc_file = "/home/haoz/data/chm1_chm13.txt" 
#fastvc_file = "/home/haoz/data/wgs_loose_goodvar.txt" 
#truth_file = "/home/old_home/haoz/workspace/data/NA12878/vcfs/NA12878_S1.vcf"
truth_file = "/home/haoz/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf"
#truth_file = "/home/haoz/data/full.37m.vcf"
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

#--- open strelka vcf file ---#
strelka_indels = dict()
if os.path.exists(sk2_file):
    with open(sk2_file, 'r') as f:
        for var in f:
            items = var.split('\t')
            if(len(items) == 10 ):
                chrom, pos, id, ref, alt, _, filter = items[:7]         
                if filter == "PASS" and (len(ref) > 1 or len(alt) > 1) :
                    #site = chrom + ":" + pos
                    site = chrom + ":" + pos + ":" + ref + ":" + alt
                    strelka_indels[site] = list([ref, alt])

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

#differ_count = 0
#same_count = 0
#for k, v in truth_indels.items():
#    if k not in strelka_indels:
#       print("not find: ", k, v)
#       differ_count += 1 
#    else:
#       if(strelka_indels[k][0] == v[0] and strelka_indels[k][1] == v[1]):
#           print("*find and equa: ", k, v)
#           same_count += 1
#       else:
#           print("find but not equa: ", k, v, "->", strelka_indels[k])
#           differ_count += 1
#
#print("truth indel num: ", len(truth_indels), "sk2 inde num", len(fastvc_indels), 
#       "differ_count", differ_count, "same_count", same_count)
#
#differ_count = 0
#same_count = 0
#for k, v in truth_indels.items():
#    if k not in fastvc_indels:
#       print("*not find: ", k, v)
#       differ_count += 1 
#    else:
#       if(fastvc_indels[k][0] == v[0] and fastvc_indels[k][1] == v[1]):
#           print("*find and equa: ", k, v)
#           same_count += 1
#       else:
#           print("*find but not equa: ", k, v, "->", fastvc_indels[k])
#           differ_count += 1
#            
#print("truth indel num: ", len(truth_indels), "fastvc inde num", len(fastvc_indels), 
#       "differ_count", differ_count, "same_count", same_count)
'''
fvc_cnt = 0
sk_cnt = 0
fvc_and_sk = 0
neither = 0
print("------------summarize--------------")
for k, v in truth_indels.items():
    if k in fastvc_indels:
        fvc_cnt += 1
        if k in strelka_indels:
            sk_cnt += 1
            fvc_and_sk += 1
    elif k in strelka_indels: 
        sk_cnt += 1
    else: 
        print("neither found: ", k, v)
        neither += 1
print("in fastvc: {}, in sk2: {}, both contained: {}, neither: {}, only fastvc: {}, only sk2: {}".format(fvc_cnt, sk_cnt, fvc_and_sk, neither, fvc_cnt - fvc_and_sk, sk_cnt - fvc_and_sk))
'''
fvc_cnt = 0
sk_cnt = 0
differ_count = 0
for k, v in truth_indels.items():
    if k in strelka_indels:
        sk_cnt += 1        
        if k not in fastvc_indels:
            print("fvc not find(sk2 find): ", k, v)
            differ_count += 1
    if k in fastvc_indels:
        fvc_cnt += 1

#print("truth indel num: ", len(truth_indels), "differ_count", differ_count)
test_total_truth = 129557
fvc_total_truth = 518228 
print("truth indel num: {}, fastvc total output: {}\nfastvc find: {} recall: {}, prec: {}"\
    .format(len(truth_indels), len(fastvc_indels), fvc_cnt, fvc_cnt / len(truth_indels), fvc_cnt / len(fastvc_indels)))
