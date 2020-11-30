#set -x 
set -e

#WEIGHT=1_$1
WEIGHT=1_10

#DATA=/home/haoz/data/CHM_data

VARTYPE="xINDEL"
if [ ${VARTYPE} == "INDEL" ]; then
    MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_germ_indel_w${WEIGHT}.pth
	DATA=/home/haoz/data/somatic/FD1018_tumoronly/FDSynthetic_tumoronly.txt
	TRUTH=/home/haoz/data/somatic/FD_10_18_data/synthetic_indels.leftAlign.vcf
    python train.py \
        --workspace /home/haoz/deepfilter/workspace \
        --train_data ${DATA} \
        --truth_file ${TRUTH} \
        --nthread 20 \
        --batch_size 8192  \
		--var_type ${VARTYPE}  \
		--weight ${WEIGHT} \
	    --out_model_path ${MODEL}
fi
VARTYPE="SNV"
if [ ${VARTYPE} == "SNV" ]; then
    MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_germ_snv_w${WEIGHT}.pth
	DATA=/home/haoz/data/somatic/FD1018_tumoronly/FDSynthetic_tumoronly.txt
	TRUTH=/home/haoz/data/somatic/FD_10_18_data/synthetic_snvs.vcf
    echo "-----------------weight info [ $WEIGHT ]----------------------"
    python train.py \
        --workspace /home/haoz/deepfilter/workspace \
        --train_data ${DATA} \
        --truth_file ${TRUTH} \
        --nthread 20 \
        --batch_size 8192 \
        --var_type ${VARTYPE} \
	    --weight ${WEIGHT} \
	    --out_model_path ${MODEL}

##    python call.py \
##        --workspace /home/haoz/deepfilter/workspace \
##        --in_data ${DATA}/test.txt \
##        --truth_file ${TRUTH} \
##        --nthread 20 \
##        --var_type ${VARTYPE} \
##	    --trained_model ${MODEL} \
##	    --out /home/haoz/deepfilter/workspace/test/germ_filtered_snv.txt

    echo "-----------------compare result------------------"
    #python compare_filter_res_snv.py 
    echo "-------------------------------------------------"
fi
##    python call.py \
##        --workspace /home/haoz/deepfilter/workspace \
##        --in_data ${DATA}/test.txt \
##        --truth_file ${TRUTH} \
##        --nthread 20 \
##        --var_type ${VARTYPE} \
##	    --trained_model ${MODEL} \
##	    --out /home/haoz/deepfilter/workspace/test/germ_filtered_indel.txt
    echo "-----------------compare result------------------"
    #python compare_filter_res_indel.py 
    echo "-------------------------------------------------"
