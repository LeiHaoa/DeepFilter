#set -x 
set -e

WEIGHT=1_$1
VARTYPE="SNV"
MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_somatic_snv_w${WEIGHT}.pth
DATA=/home/haoz/data/somatic/FD_10_18_data
echo "-----------------weight info [ $WEIGHT ]----------------------"
python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data ${DATA}/train1.txt \
    --truth_file ${DATA}/synthetic_snvs.vcf \
    --nthread 20 \
    --batch_size 1024 \
    --var_type ${VARTYPE} \
	--weight ${WEIGHT} \
	--out_model_path ${MODEL}

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data ${DATA}/train2.txt \
    --truth_file ${DATA}/synthetic_snvs.vcf \
	--pretrained_model ${MODEL} \
    --nthread 20 \
    --batch_size 1024 \
    --var_type ${VARTYPE} \
	--weight ${WEIGHT} \
	--out_model_path ${MODEL}

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data ${DATA}/train3.txt \
    --truth_file ${DATA}/synthetic_snvs.vcf \
	--pretrained_model ${MODEL} \
    --nthread 20 \
    --batch_size 1024 \
    --var_type ${VARTYPE} \
	--weight ${WEIGHT} \
	--out_model_path ${MODEL}

python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${DATA}/test.txt \
    --truth_file ${DATA}/synthetic_snvs.vcf \
    --nthread 20 \
    --var_type ${VARTYPE} \
	--trained_model ${MODEL} \
	--out /home/haoz/deepfilter/workspace/test/filtered_snv.txt

echo "-----------------compare result------------------"
#python compare_filter_res_snv.py 
python compare_filter_res_som_snv.py 
echo "-------------------------------------------------"
