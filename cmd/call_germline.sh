set -e 

# description: bash script of calling germline variants(snv and indel)

DATA=/home/haoz/data/somatic/FD1021_tumoronly
THREAD=20
IN_DATA=${DATA}/FDSynthetic1021_tumoronly.txt
DEEPFILTER=/home/haoz/deepfilter

#-----------------call indel-----------------#
VARTYPE="xINDEL"
INDEL_MODEL=${DEEPFILTER}/workspace/test/checkpoint_germ_indel_w1_100.pth
if [ ${VARTYPE} == "INDEL" ]; then
time python call.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${IN_DATA} \
    --truth_file ./empty.vcf \
    --nthread ${THREAD} \
    --var_type ${VARTYPE} \
	--trained_model ${INDEL_MODEL} \
	--out /home/haoz/deepfilter/workspace/result/filtered_indel.txt

python compare_filter_res_indel.py 
fi
#-----------------call snv-----------------#
VARTYPE="SNV"
SNV_MODEL=/home/haoz/deepfilter/workspace/test/checkpoint_germ_snv_w1_10.pth

time python call.py \
    --workspace /home/haoz/deepfilter/workspace \
    --in_data ${IN_DATA} \
    --truth_file ./empty.vcf \
    --nthread ${THREAD} \
    --var_type ${VARTYPE} \
	--trained_model ${SNV_MODEL} \
	--out /home/haoz/deepfilter/workspace/result/filtered_snv.txt

echo "--------------------start checking------------------"
python compare_filter_res_snv.py 

#cd ${DEEPFILTER}/workspace/result
#RES_NAME=trainset1021_tumoronly
#if [ -f ${RES_NAME}.txt ]; then
#	rm ${RES_NAME}.txt
#fi
#touch ${RES_NAME}.txt
#cat filtered_indel.txt >> ${RES_NAME}.txt
#cat filtered_snv.txt >> ${RES_NAME}.txt
##rm -f filtered_indel.txt
#
#cat ./${RES_NAME}.txt | ${DEEPFILTER}/var2vcf_valid.pl -A > ${RES_NAME}.vcf
#if [ -f ${RES_NAME}.vcf.gz ]; then
#		echo "file ${RES_NAME}.vcf.gz exist! will delete it"
#		rm -f ${RES_NAME}.vcf.gz
#		rm -f ${RES_NAME}.vcf.gz.tbi
#fi
#bgzip ${RES_NAME}.vcf ${RES_NAME}.vcf.gz && tabix -p vcf ${RES_NAME}.vcf.gz
#
#echo "---------------starting hap.py....--------------"
#time som.py \
#	/home/haoz/data/somatic/FD_10_21_data/synthetic_indels.leftAlign.vcf
#	${RES_NAME}.vcf.gz \
#	-o trainset1021.tumoronly.indel \
#	-r ${dataset}/hg38/hg38.fa \
#	--verbose
#
#time python2 som.py \
#	/home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf
#	${RES_NAME}.vcf.gz \
#	-o trainset1021.tumoronly.snv \
#	-r ${dataset}/hg38/hg38.fa \
#	--verbose
#
#	#/home/haoz/data/somatic/FD1021_tumoronly/FDSynthetic1021_tumoronly.txt \
