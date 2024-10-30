# 1. for alignment annotation
write_rmsk_intron_bed_script = """
#!/bin/bash

REP_BED=$1
GFF3=$2
ALU_MERGE_DIST=$3
TMPDIR=$4
CHROM=$5


cd $TMPDIR
## file setup
exon_bed="${CHROM}.exon.bed"
gene_bed="${CHROM}.gene.bed"
gene_slop_bed="${CHROM}.gene_slop.bed"
intron_bed="${CHROM}.intron.bed"
anno_rep_bed="${CHROM}.anno_repeat.bed"

# 1. sort and cut repeatMasker bed
cat ${REP_BED} | gawk -F "\\t" -vOFS="\\t" \\
  '{print $1,$2,$3,$4"|"$8"|"$7,$6}' | \\
  sort -k2,2n -k3,3n > ${REP_BED}.sorted
mv ${REP_BED}.sorted ${REP_BED} # for site annotation as well

# 2. process gff3 file to get exon, gene, and gene_slop
sort -k2,2n -k3,3n ${GFF3} | uniq > ${GFF3}.sorted
mv ${GFF3}.sorted ${GFF3}

cat ${GFF3} | gawk -F"\\t" -vOFS="\\t" '
{
  if ($3 == "exon") {
    print $1, $4, $5 >> "'$exon_bed'"
  } 
  else if ($3 == "gene") {
    split($9, attributes, ";")
    gname = "NA"
    for (i in attributes) {
      if (attributes[i] ~ /^.?gene_id/ || attributes[i] ~ /^gene_id=/) {
        split(attributes[i], a, "[=|\\"]")
        gid = a[2]
        break
      }
    }
    print $1, $4, $5, gid >> "'$gene_bed'"
    # Additionally, calculate the "slop" here and print to another file
    print $1, ($4-1000 < 0 ? 0 : $4-1000), $5+1000, gid >> "'$gene_slop_bed'"
  }
}'

### for chr1,2,... use -k1,1V ###
sort -k2,2n -k3,3n ${exon_bed} | uniq > ${exon_bed}.sorted
mv ${exon_bed}.sorted ${exon_bed}

sort -k2,2n -k3,3n ${gene_bed} | uniq > ${gene_bed}.sorted
mv ${gene_bed}.sorted ${gene_bed}

sort -k2,2n -k3,3n ${gene_slop_bed} | uniq > ${gene_slop_bed}.sorted
mv ${gene_slop_bed}.sorted ${gene_slop_bed}

# 3. subtract exon from gene to get intron
bedtools merge -i ${exon_bed} | \\
  bedtools subtract -a ${gene_bed} -b - | \\
  gawk -F"\\t" -vOFS="\\t" '{$4="intron"; print $0}' | \\
  gawk '!seen[$0]++' | \\
  bedtools intersect -a - -b ${gene_bed} -wao | \\
  gawk -F"\\t" -vOFS="\\t" '!seen[$1,$2,$3]++ && $9 >= 3 {
    print $1,$2,$3,$4,".",$8
  }' > ${intron_bed}

# 4. get integenic Alu locus coordinates, set distance to 500 (default)
cat ${REP_BED} | gawk '$4 ~ /Alu/' | \\
  bedtools merge -i - -d ${ALU_MERGE_DIST} | \\
  bedtools subtract -a - -b ${gene_slop_bed} | \\
  gawk '($3-$2) >= 5 {
    print $0"\\tAlu."$1":"$2"-"$3
  }' | cat ${gene_slop_bed} - | \\
  sort -k2,2n -k3,3n | \\
  bedtools intersect -a ${REP_BED} -b - -wao | cut -f 1-5,9,10 | \\
  sort -k2,2n -k3,3n -nrk7,7 | \\
  gawk '!seen[$1,$2,$3]++' | cut -f1-6 > ${anno_rep_bed}

# 5. split by chromosome
cat ${anno_rep_bed} ${intron_bed} | sort -k2,2n -k3,3n | \\
    gawk -F'\\t' -vOFS='\\t' '{c=$1; print >> c".repeat.intron.bed"}'

rm ${anno_rep_bed} ${intron_bed} ${gene_bed} ${gene_slop_bed} ${exon_bed}
"""


# 2. for mismatch annotation using gff3/gtf and repeatmasker
## repeat and gff3 bed are already split by chromosome
site_anno_script = """
#!/bin/bash

SITE_BED=$1
REP_INTRON_BED=$2
GFF3_BED=$3
TMPDIR=$4
CHROM=$5

cd $TMPDIR
sort -k2,2n ${SITE_BED} | uniq > ${SITE_BED}.sorted
mv ${SITE_BED}.sorted ${SITE_BED}

# 1. repeatMasker bed
bedtools intersect -a ${SITE_BED} -b ${REP_INTRON_BED} -wa -wb | cut -f1-3,7-9 | \\
  gawk '!seen[$0]++' > ${CHROM}.site.repeat.intron.bed

# 2. gff3 bed
cat ${GFF3_BED} | gawk '!seen[$0]++' | \\
  sort -k2,2n -k3,3n | \\
  bedtools intersect -a ${SITE_BED} -b - -wa -wb | \\
  cut -f1-3,7-9 | gawk '!seen[$0]++' > ${CHROM}.site.gff3.bed
"""
