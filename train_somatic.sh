set -x 
set -e

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data /home/haoz/data/somatic/FD_10_18_data/train1.txt \
    --truth_file /home/haoz/data/somatic/FD_10_18_data/synthetic_snvs.vcf \
    --nthread 20 \
    --var_type "SNV" \
	--weight 1_10 \
	--out_model_path /home/haoz/deepfilter/workspace/test/checkpoint_snv_w1_10.pth
	#--pretrained_model /home/haoz/deepfilter/workspace/models/checkpoint_INDEL_20-10-20-20-22-45_ecpch10.pth \
