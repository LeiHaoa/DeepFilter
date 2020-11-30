set -x 
set -e
DEEPFILTER=/home/haoz/deepfilter
cd ${DEEPFILTER}/workspace/result
RES_NAME=trainset1021_tumoronly

echo "---------------starting hap.py....--------------"
time som.py \
	/home/haoz/data/somatic/FD_10_21_data/synthetic_indels.leftAlign.vcf
	${RES_NAME}.vcf.gz \
	-o trainset1021.tumoronly.indel \
	-r ${dataset}/hg38/hg38.fa \
	--verbose

time som.py \
	/home/haoz/data/somatic/FD_10_21_data/synthetic_snvs.vcf
	${RES_NAME}.vcf.gz \
	-o trainset1021.tumoronly.snv \
	-r ${dataset}/hg38/hg38.fa \
	--verbose
