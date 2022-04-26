# Data Instruction

## SEQC-II
- [data availability](https://sites.google.com/view/seqc2/home/sequencing)
- Alignment
``` bash
if [ ! -f "${FD_SOURCE}/${SAMPLE}.sorted.bam" ]; then
  echo "you do not have sorted bam data"
  time $bwa mem -t 20 -R "@RG\tID:${SAMPLE}\tSM:${SAMPLE}\tLB:WGS\tPL:Illumina" ${REFERNCE} \
    ${FASTQ_FILES}_1.fastq ${FASTQ_FILES}_2.fastq | samtools view -Sb - > ${FD_SOURCE}/${SAMPLE}.bam
  echo "prepare done"
  TMP=${FD_SOURCE}/${SAMPLE}
  samtools sort ${TMP}.bam -o ${TMP}.sorted.bam
  samtools index ${TMP}.sorted.bam
  echo "done"
  rm -f $TMP.bam
  rm -f ${FASTQ_FILES}_*.fastq
else
  echo "find normal bam file, continue..."
fi
```
- Detection by VarDict
``` bash
VarDict -G /path/to/hg19.fa -f 0.001 -N "tumor|normal" -b "/path/to/tumor.bam|/path/to/normal.bam"  wgs_hg38.bed --filsher > vars.txt
```

## SYNC DATA

- BamSurgen script
1. Downoad the Docker image
``` bash
docker pull lethalfang/bamsurgeon
```
2. generate scripts
```
SOMATICSEQ_PATH=/home/user_home/haoz/git/somaticseq
${SOMATICSEQ_PATH}/somaticseq/utilities/dockered_pipelines/bamSimulator/BamSimulator_multiThreads.sh \
--genome-reference  ${REFERNCE} \
--tumor-bam-in      ${FD_SOURCE}/FD2_T.sorted.bam \
--normal-bam-in     ${FD_SOURCE}/FD2_N.sorted.bam \
--tumor-bam-out     syntheticTumor.bam \
--normal-bam-out    syntheticNormal.bam \
--threads 40 \
--split-proportion  0.5 \
--num-snvs          300000 \
--num-indels        100000 \
--min-vaf           0.001 \
--max-vaf           1.0 \
--left-beta         2 \
--right-beta        5 \
--min-variant-reads 2 \
--output-dir        /home/user_home/haoz/data/HCC1395_sync/Sim_0324 \
--seed        1024 \
--merge-output-bams
```
3. run in parallel
```
parall -j 40 < jobs.jb
```
jobs.jb list the .cmd file output by step2.
4. run merge
run the merge file (mergeFiles.date-xxx.cmd) output by step2
```bash
bash /xxx/xxx/mergeFiles.cmd
```