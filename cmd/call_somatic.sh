#set -x 
set -e

WEIGHT=1_100
VARTYPE="SNV"
#MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_snv_w${WEIGHT}.pth
MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_w${WEIGHT}.pth
DATA=/home/haoz/data/somatic/B170NC
if [ ! -f ./empty.vcf ]; then
	touch ./empty.vcf
fi

python call_somatic.py \
	--workspace /home/haoz/deepfilter/workspace \
	--in_data /home/haoz/data/somatic/FD_10_21_data/test.txt \
	--truth_file ./empty.vcf\
	--nthread 20 \
	--var_type "SNV" \
	--trained_model /home/haoz/deepfilter/workspace/test/exp_adam_somatic_snv_trainset1018_w1_2.pth.1999 \
	--out /home/haoz/deepfilter/workspace/test/filtered_snv.txt

#echo "-----------------compare result------------------"
python compare_filter_res_som_snv.py 
#echo "-------------------------------------------------"
