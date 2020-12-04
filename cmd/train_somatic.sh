set -x 
set -e

ROUND=$1
WEIGHT=1_2

TRAIN_DATA=/home/haoz/data/somatic/FD_DATASET_1/FD_DATA_1.txt
SNV_TRUTH_FILE=/home/haoz/data/somatic/FD_DATASET_1/FDtruth_Data_1.snv.vcf
INDEL_TRUTH_FILE=/home/haoz/data/somatic/FD_DATASET_1/FDtruth_Data_1.indel.vcf

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data ${TRAIN_DATA} \
	--truth_file ${SNV_TRUTH_FILE} \
	--batch_size 2048 \
    --nthread 20 \
    --var_type "SNV" \
	--weight ${WEIGHT} \
	--out_model_path /home/haoz/deepfilter/workspace/test/somatic_snv_fd1_w1_2_round${ROUND}.pth

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data ${TRAIN_DATA} \
	--truth_file ${INDEL_TRUTH_FILE} \
	--batch_size 1024 \
    --nthread 20 \
    --var_type "INDEL" \
	--weight ${WEIGHT} \
	--out_model_path /home/haoz/deepfilter/workspace/test/somatic_indel_fd1_w1_2_round${ROUND}.pth

#python call_somatic.py \
#    --workspace /home/haoz/deepfilter/workspace \
#    --in_data /home/haoz/data/somatic/FD_10_21_data/test.txt \
#    --truth_file /home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf \
#    --nthread 20 \
#    --var_type "SNV" \
#	--trained_model /home/haoz/deepfilter/workspace/test/exp_adj_somatic_snv_trainset1018_w${WEIGHT}.pth \
#	--out /home/haoz/deepfilter/workspace/test/filtered_snv.txt
#
#	#--trained_model /home/haoz/deepfilter/workspace/test/somatic_snv_trainset1018_w1_10.pth \
#	#--trained_model /home/haoz/deepfilter/workspace/test/checkpoint_somatic_snv_w1_40.pth \
#
# python call_somatic.py \
#     --workspace /home/haoz/deepfilter/workspace \
#     --in_data /home/haoz/data/somatic/FD_10_21_data/test.txt \
#     --truth_file /home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf \
#     --nthread 20 \
#     --var_type "INDEL" \
# 	--trained_model /home/haoz/deepfilter/workspace/test/somatic_indel_trainset1018_w${WEIGHT}.pth \
# 	--out /home/haoz/deepfilter/workspace/test/filtered_indel.txt

#echo "-----------------compare result------------------"
#python compare_filter_res_som_snv.py
#python compare_filter_res_som_indel.py
#echo "-------------------------------------------------"
