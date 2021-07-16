set -x
set -e

weight=$1

python train_somatic.py \
  --workspace /home/haoz/RabbitVar/DeepFilter \
  --train_data  /home/haoz/RabbitVar/DeepFilter/train_data/data_indel_all.tsv \
  --nthread 8 \
  --var_type "INDEL" \
  --weight ${weight} \
  --out_model_path /home/haoz/RabbitVar/DeepFilter/models/checkpoint_indel_w${weight}.adam.pth
