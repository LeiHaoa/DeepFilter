set -e 

DATA=/home/haoz/data/somatic/B170NC
THREAD=20
IN_DATA=${DATA}/B1701_B17NC.notloose.txt \
#-----------------call indel-----------------#
WEIGHT=1_100
VARTYPE="INDEL"
#MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_snv_w${WEIGHT}.pth
INDEL_MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_w${WEIGHT}.pth

python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${IN_DATA} \
    --truth_file ${DATA}/../FD_10_18_data/synthetic_snvs.vcf \
    --nthread ${THREAD} \
    --var_type ${VARTYPE} \
	--trained_model ${MODEL} \
	--out /home/haoz/deepfilter/workspace/test/filtered_indel.txt

#-----------------call snv-----------------#
WEIGHT=1_100
VARTYPE="SNV"
SNV_MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_snv_w${WEIGHT}.pth

python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${IN_DATA} \
    --truth_file ${DATA}/../FD_10_18_data/synthetic_snvs.vcf \
    --nthread ${THREAD} \
    --var_type ${VARTYPE} \
	--trained_model ${MODEL} \
	--out /home/haoz/deepfilter/workspace/test/filtered_snv.txt
