set -x 
set -e

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data /home/haoz/data/somatic/FDSynthetic.notloose.txt \
    --truth_file /home/haoz/data/somatic/synthetic_indels.leftAlign.vcf \
	--pretrained_model /home/haoz/deepfilter/workspace/models/checkpoint_INDEL_20-10-20-20-22-45_ecpch10.pth \
    --nthread 20 \
    --var_type "INDEL" \
