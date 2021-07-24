# DeepFilter

rabbitvar filter

A deep-learning-based variant filter for VarDict

## An example using DeepFilter to filter INDEL or SNV variants

- Filter INDEL variant and then format to VCF file
```
VARTYPE="INDEL"
python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${IN_DATA} \
    --nthread ${THREAD} \
    --var_type ${VARTYPE} \
	--trained_model ./models/checkpoint_indel_w1_24.adam.pth \
	--out ${DEEPFILTER}/workspace/result/filtered_som_indel.vcf
```

- Filter SNV variant and then format to VCF file
```
VARTYPE="INDEL"
python call_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${IN_DATA} \
    --nthread ${THREAD} \
    --var_type ${VARTYPE} \
	--trained_model ./models/checkpoint_snv_w1_24.adam.pth \
	--out ${DEEPFILTER}/workspace/result/filtered_som_snv.vcf
```

## Train new models

- step1: make .csv data
```
python make_data.py inter.txt groundtruth.vcf $TYPE train_data.tsv
```
- step2: modified the source code if you want to change the structure of the network or other strategies.

- step3: re-train the model

```
python train_somatic.py \
  --workspace . \
  --train_data  ${data_path}/data_indel_all.tsv \
  --nthread 8 \
  --var_type "INDEL" \
  --weight ${weight} \
  --out_model_path ./models/checkpoint_indel_w${weight}.adam.pth
```

# Usage

```
sage: call_somatic.py [-h] --workspace WORKSPACE 
                      --in_data IN_DATA --truth_file TRUTH_FILE
                      [--model_out MODEL_OUT] --var_type VAR_TYPE
                      [--batch_size BATCH_SIZE] [--nthreads NTHREADS]
                      [--trained_model TRAINED_MODEL] --out OUT
```
