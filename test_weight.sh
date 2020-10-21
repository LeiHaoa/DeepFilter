#set -x 
set -e

#$1 = 1_10
WEIGHT=1_$1
MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_snv_w${WEIGHT}.pth
echo "-----------------weight info [ $WEIGHT ]----------------------"
python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data /home/haoz/data/somatic/FD_10_18_data/train.txt \
    --truth_file /home/haoz/data/somatic/FD_10_18_data/synthetic_snvs.vcf \
    --nthread 20 \
    --var_type "SNV" \
	--weight ${WEIGHT} \
	--out_model_path ${MODEL}

python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data /home/haoz/data/somatic/FD_10_18_data/test.txt \
    --truth_file /home/haoz/data/somatic/FD_10_18_data/synthetic_snvs.vcf \
    --nthread 20 \
    --var_type "SNV" \
	--trained_model ${MODEL} \
	--out /home/haoz/deepfilter/workspace/test/filtered_snv.txt

echo "-----------------compare result------------------"
python compare_filter_res_snv.py 
echo "-------------------------------------------------"
