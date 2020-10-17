set -x 
set -e

python train_somatic.py \
    --workspace /home/haoz/deepfilter/workspace \
    --train_data /home/haoz/data/somatic/HCC1187C_HCC1187BL_raw.txt \
    --truth_file /home/haoz/data/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_PGandRTGphasetransfer.vcf \
    --nthread 20 \
    --var_type "INDEL" \
