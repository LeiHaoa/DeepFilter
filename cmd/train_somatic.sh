set -x 
set -e

TYPE=$1
WEIGHT=1_2
if [ ${TYPE} == "train_snv" ]; then
python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data /home/haoz/data/somatic/FD_10_18_data/FDSynthetic.notloose.txt \
    --truth_file /home/haoz/data/somatic/FD_10_18_data/synthetic_snvs.vcf \
	--batch_size 2048 \
    --nthread 20 \
    --var_type "SNV" \
	--weight ${WEIGHT} \
	--out_model_path /home/haoz/deepfilter/workspace/test/exp_adam_somatic_snv_trainset1018_w${WEIGHT}.pth

elif [ ${TYPE} == "train_indel" ]; then
python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data /home/haoz/data/somatic/FD_10_18_data/FDSynthetic.notloose.txt \
    --truth_file /home/haoz/data/somatic/FD_10_18_data/synthetic_indels.leftAlign.vcf \
	--batch_size 1024 \
    --nthread 20 \
    --var_type "INDEL" \
	--weight ${WEIGHT} \
	--out_model_path /home/haoz/deepfilter/workspace/test/somatic_indel_trainset1018_w${WEIGHT}.pth
else
	echo "you are training nothing, call directly"
fi

python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data /home/haoz/data/somatic/FD_10_21_data/test.txt \
    --truth_file /home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf \
    --nthread 20 \
    --var_type "SNV" \
	--trained_model /home/haoz/deepfilter/workspace/test/exp_adj_somatic_snv_trainset1018_w${WEIGHT}.pth \
	--out /home/haoz/deepfilter/workspace/test/filtered_snv.txt

	#--trained_model /home/haoz/deepfilter/workspace/test/somatic_snv_trainset1018_w1_10.pth \
	#--trained_model /home/haoz/deepfilter/workspace/test/checkpoint_somatic_snv_w1_40.pth \

# python call_somatic.py \
#     --workspace /home/haoz/deepfilter/workspace \
#     --in_data /home/haoz/data/somatic/FD_10_21_data/test.txt \
#     --truth_file /home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf \
#     --nthread 20 \
#     --var_type "INDEL" \
# 	--trained_model /home/haoz/deepfilter/workspace/test/somatic_indel_trainset1018_w${WEIGHT}.pth \
# 	--out /home/haoz/deepfilter/workspace/test/filtered_indel.txt

echo "-----------------compare result------------------"
python compare_filter_res_som_snv.py
python compare_filter_res_som_indel.py
echo "-------------------------------------------------"
