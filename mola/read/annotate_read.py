from mola.read.process_alignment import *
from mola.parse.parse_gff import fetch_gene_names, gff2bed
from mola.parse.parse_bed import write_rmsk_intron_bed_script

#####################################################
# Modes:
#  * bulk:
#   - single end short reads
#   - paired end short reads 
#    (keep R1 and R2 in objects for mismatch calling, 
#     only count one read in counting)
#  * single cell:
#   - 10x 3'/5' short reads
#   - 10x 3'/5' long reads
#####################################################

def annotate_reads(
        bam_path,
        bulk, 
        paired_end,
        primary,
        min_len,
        min_mapq,
        read_assignments_path,
        gff3_path,
        repeat_bed_path,
        alu_merge_dist,
        min_overlap,
        min_exon_on_read, 
        min_repeat_on_read,
        min_intron_on_read,
        min_intron_unspliced,
        write_read_info,
        num_threads,
        startswith_chr,
        tmp_dir,
        out_dir
    ):

    if not futils.is_command_avail('bedtools'):
        logging.error(
            'Make sure bedtools is installed, try "conda install -c bioconda bedtools"'
        )
        sys.exit(1)
    if platform.system() == 'Darwin' and not futils.is_command_avail('gawk'):
        logging.error(
            'Make sure gawk is installed, try "brew install gawk"'
        )
        sys.exit(1)

    futils.startswith_chr = startswith_chr
    logging.info('Loading and processing inputs...')
    read_info = digest_read_assignment(read_assignments_path)

    process_anno = False if tmp_dir is not None else True
    read_dir, tmp_dir = output_setup(out_dir, tmp_dir) # tmp_dir gets updated
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        chrom_list = list(bam.references)
    if process_anno:
        split_anno_files(chrom_list, tmp_dir, gff3_path, repeat_bed_path)
    chrom_list = update_chrom_list(chrom_list, tmp_dir)
    
    # run on each chromosome
    logging.info('Start annotating reads...')
    _ = futils.process_chrom_in_threads(
        process_chromosome, 
        chrom_list,
        num_threads,
        bam_path=bam_path,
        bulk=bulk,
        paired_end=paired_end,
        primary=primary,
        min_len=min_len,
        min_mapq=min_mapq,
        read_info=read_info,
        min_overlap=min_overlap,
        min_exon_on_read=min_exon_on_read,
        min_repeat_on_read=min_repeat_on_read,
        min_intron_on_read=min_intron_on_read,
        min_intron_unspliced=min_intron_unspliced,
        write_read_info=write_read_info,
        alu_merge_dist=alu_merge_dist,
        read_dir=read_dir,
        tmp_dir=tmp_dir,
        process_anno=process_anno
    )
    return tmp_dir

def output_setup(out_dir, tmp_dir):
    read_obj_dir = os.path.join(os.path.expanduser(out_dir), 'objects', 'reads')
    os.makedirs(read_obj_dir, exist_ok=True)
    
    if not tmp_dir:
        tmp_dir = futils.make_tmp_dir(out_dir=out_dir, suffix='_beds')
    else:
        tmp_dir = os.path.abspath(os.path.expanduser(tmp_dir))
    return read_obj_dir, tmp_dir

def update_chrom_list(chrom_list, tmp_dir):
    # List all files in the directory
    anno_files = os.listdir(tmp_dir)
    chrom_ids = [f.split('.')[0] for f in anno_files if f.endswith('.gff3')]
    return [c for c in chrom_list if c in chrom_ids]

#GOOD_CAT = ['unique', 'unique_minor_difference', 'ambiguous', 'inconsistent']
BAD_CAT = ['noninformative', 'intergenic']
def digest_read_assignment(read_assignment_path, bad_categories=BAD_CAT, good_read=True):
    '''
    read_assignment (FORMAT FIXED, basically one of the outputs from isoquant): 
    read  chr  strand  isoform  gene etc
    '''
    if not read_assignment_path:
        logging.info('Read assignments not provided, assume short reads. Features from genes will be prioritized over repeats.')
        logging.info('Gene and strand assignments from tags/annotation and alignment.')
        return None
    
    read_info = defaultdict(dict)
    with futils.open_text(read_assignment_path) as fin:
        for line in fin:
            l = futils.read_line(line)
            if l[0].startswith('#') or (l[0] in read_info and read_info[l[0]]['g'] != '.'):
                continue
            
            read, chrom, strand, tx, gene, category = l[:6]
            if good_read and category in bad_categories:
                # strand can be . (unknown)
                continue
            
            gene = '.' if not gene else gene
            if 't' in read_info[read] and tx not in read_info[read]['t']:
                read_info[read]['t'].append(tx)
            else:
                read_info[read] = {'c': chrom, 's': strand, 'g': gene, 't': [tx]}
    return read_info

def split_anno_files(chrom_list, tmp_dir, gff3_path, repeat_bed_path):
    sep_file_by_chrom(chrom_list, tmp_dir, gff3_path, 'gff3')
    sep_file_by_chrom(chrom_list, tmp_dir, repeat_bed_path, 'repeat.bed')
    # print out the bash script to process annotations for alignments
    futils.write_base_script(write_rmsk_intron_bed_script, f'{tmp_dir}/write_bed.sh')

def sep_file_by_chrom(chrom_list, tmp_dir, file, suffix):
    file_handles = {}
    try:
        with futils.open_text(file) as f:
            for line in f:
                l = futils.read_line(line, sep='\t')
                if not l[0].startswith('#'):
                    break
                
            for line in itertools.chain([line], f):
                l = futils.read_line(line)
                chrom = futils.check_chr_id(l[0])
                if chrom not in chrom_list:
                    continue
                if chrom not in file_handles:
                    out_file = os.path.join(os.path.expanduser(tmp_dir), f'{chrom}.{suffix}')
                    file_handles[chrom] = futils.write_text(out_file)
                
                file_handles[chrom].write(futils.list2line(l))
    finally:
        [fh.close() for fh in file_handles.values()]

def process_chromosome(chrom, bam_path, bulk, paired_end, primary, min_len, min_mapq, 
                    read_info, min_overlap, min_exon_on_read, min_repeat_on_read, 
                    min_intron_on_read, min_intron_unspliced, write_read_info, alu_merge_dist, 
                    read_dir, tmp_dir, process_anno):
    mode = 'bulk' if bulk else 'sc'
    mode = f'{mode}-pe' if paired_end else f'{mode}-se'
    logging.info(f'[{mode}] {chrom}')
    # process annotation files
    if process_anno:
        write_anno_beds(alu_merge_dist, tmp_dir, chrom)
    # write read infor file if specified
    read_info = None
    if write_read_info:
        read_info_path = f'{tmp_dir}/{chrom}.read_info.tsv.gz'
        read_info = futils.write_text(read_info_path)
        header = ['barcode', 'umi', 'read', 'length', 'strand', 'feature', 'feature_id', 'feature_name']
        finfo.write(futils.list2line(header).encode())
        
    # TODO - add exception handling for unmatched chromosomes between BAM & annotations
    bed_out = f'{tmp_dir}/{chrom}.aln.bed'
    confident_gene = True if read_info is not None else False
    gff3_path = f'{tmp_dir}/{chrom}.gff3'
    geneid_to_name = fetch_gene_names(gff3_path)

    with pysam.AlignmentFile(bam_path) as bam:
        bam_chrom = bam.fetch(chrom)
        reads, dup_cnt = iterate_bam(
            bam_chrom, bulk, primary, min_len, min_mapq, 
            read_info, geneid_to_name, bed_out
        )
    
    # generate exon bed from gff3/gtf
    gff2bed(f'{tmp_dir}/{chrom}.gff3', f'{tmp_dir}/{chrom}.gff3.bed')
    exec_intersect(tmp_dir, chrom)

    rep_int_intersect = f'{tmp_dir}/{chrom}.rep.int.intsc.bed.gz'
    exon_intersect = f'{tmp_dir}/{chrom}.exon.intsc.bed.gz' # if mode=='bulk-pe' else None
    reads = classify_reads(
        rep_int_intersect, 
        exon_intersect, 
        paired_end,
        min_overlap, 
        min_exon_on_read,
        min_repeat_on_read, 
        min_intron_on_read,
        min_intron_unspliced,
        read_info, 
        confident_gene, 
        reads, 
        geneid_to_name
    )
    if read_info:
        read_info.close()
    futils.save_gz_pickle(f'{read_dir}/{chrom}_reads.pkl.gz', reads)
    if bulk:
        logging.info(f'[{mode}] {chrom} done! {len(reads)} molecules.')
    else:
        logging.info(f'[{mode}] {chrom} done! {dup_cnt} PCR duplicates discarded; {len(reads)} unique molecules left.')

def write_anno_beds(alu_merge_dist, tmp_dir, chrom):
    bash_path = f'{tmp_dir}/write_bed.sh'
    rep_bed = f'{tmp_dir}/{chrom}.repeat.bed'
    gff3 = f'{tmp_dir}/{chrom}.gff3'
    cmd = ['bash', bash_path, rep_bed, gff3, alu_merge_dist, tmp_dir, chrom]
    subprocess.run(cmd, check=True)

def iterate_bam(bam, bulk, primary, min_len, min_mapq, read_info,
                geneid_to_name, bed_out):
    '''
    Iterate bam file and initial Read objects by chromosome.
        - For coding reads, get strand and gene info from read_assignment;
        - For noncoding reads, get strand and annotation from annotation bed (eg repeatMasker etc),
            therefore when iterating bam, convert noncoding reads into bed file
    '''
    reads_chrom = dict()
    uniq_molecules = dict() # unique cell barcode + umi pairs
    dup_cnt = 0
    with futils.write_text(bed_out) as fout:
        for aln in bam:
            read = aln2Read(aln, primary, min_len, min_mapq)
            if not read:
                continue

            if bulk:
                # some paired end reads don't have paired ids
                if aln.query_name in reads_chrom:
                    read.id = f'{read.id}/2'
            elif dup_mole(read, uniq_molecules, max_dist=1):
                dup_cnt += 1
                continue

            read.feature = '.' # marker for annotated reads
            if not read_info:
                # short reads don't have confident gene assignments to reads
                read.strand = '-' if aln.is_reverse else '+'
                if not bulk:
                    # NOTE gene tags from cellranger may not be correct as they are from STAR
                    # however will trust genic/non-genic assignments
                    gene_id = get_tags(aln, ['GX'])
                    if not gene_id:
                        gene_id = get_tags(aln, ['AN'])
                        # AN = antisense, means read strand is opposite to gene strand
                        if gene_id:
                            gene_id = random.choice(gene_id.split(','))
                            read.feature = 'an'
                        else:
                            reads_chrom[read.id] = read
                            write_read_bed(read, aln, fout)
                            continue
                    else:
                        # regardless of xf score as it focuses on counting genes
                        read.feature = 'gene'

                    read.gene['id'] = gene_id
                    if gene_id in geneid_to_name:
                        gname = geneid_to_name[gene_id]
                    else:
                        gname = get_tags(aln, 'GN')
                        gname = gname if gname is not None else '.'
                    read.gene['name'] = ';'.join(np.unique(gname.split(';'))) # prevent repeated names
            elif read.id in read_info:
                # long reads have confident gene assignments to reads
                rinfo = read_info[read.id]
                gene_chrom, strand, gene_id, tx_id = rinfo['c'], rinfo['s'], rinfo['g'], rinfo['t']
                if gene_chrom != read.chr:
                    logging.debug(f'{read.id} from {read.chr} but was assigned to {gene_chrom}')
                    continue
                
                gname, gstrand = geneid_to_name[gene_id].split('|')
                read.feature = 'gene'
                read.strand = strand if strand != '.' else gstrand # splice site strand > annotation strand
                read.gene['id'] = gene_id
                read.gene['name'] = geneid_to_name[gene_id]
                read.gene['tx'] = random.choice(tx_id) # randomly picked one!
                read.struct = 'S' # preset to spliced for all genic reads
            reads_chrom[read.id] = read
            write_read_bed(read, aln, fout) # get bed file for all (short or long) reads
    return reads_chrom, dup_cnt

def dup_mole(read, uniq_molecules, max_dist):
    # is_duplicate flag seems useless, dedup here
    cb, umi, pos = read.cb, read.umi, read.start
    if (cb, umi) not in uniq_molecules:
        uniq_molecules[(cb, umi)] = set([pos])
    else:
        all_pos = uniq_molecules[(cb, umi)]
        for pos in all_pos:
            if pos in range(pos-max_dist, pos+max_dist+1):
                return True
    return False

def write_read_bed(read, aln, fout):
    other_info = [read.len, read.id, read.gene.get('id', '.'), read.gene.get('tx', '.')]
    bed_rows = read.write_bed(aln.cigartuples, other_info)
    for r in bed_rows:
        fout.write(futils.list2line(r))

def exec_intersect(tmp_dir, chrom):
    aln_bed = f'{tmp_dir}/{chrom}.aln.bed'
    # intersect with intronic & intergenic (repeat) regions to get repeat annotation
    rep_intron_bed = f'{tmp_dir}/{chrom}.repeat.intron.bed'
    rep_intron_out = f'{tmp_dir}/{chrom}.rep.int.intsc.bed.gz'
    os.system(f'bedtools intersect -a {aln_bed} -b {rep_intron_bed} -wo | gzip > {rep_intron_out}')
    
    # intersect with gene exon regions to get exonic annotation 
    # get gene annotation for bulk short reads (usually paired end)
    exon_bed = f'{tmp_dir}/{chrom}.gff3.bed'
    exon_out = f'{tmp_dir}/{chrom}.exon.intsc.bed.gz'
    os.system(f'bedtools intersect -a {aln_bed} -b {exon_bed} -wo | cut -f1-15,17 | gzip > {exon_out}')

def classify_reads(rep_int_intersect, exon_intersect, paired_end, 
                    min_overlap, min_exon_on_read, min_repeat_on_read, min_intron_on_read, 
                    min_intron_unspliced, read_info, confident_gene, reads, geneid_to_name):
    '''
    - Categories (repeat reads are messy af, three categories are enough for now):
        - gene
        - Alu
        - Alu + other repeat
        - other repeat
    - min_overlap: minimum overlap threshold for the shorter one
    - min_exon_on_read: minimum exon ratio in the read to call it a exonic read (for short reads)
    - min_repeat_on_read: minimum repeat ratio in the read to call it a repeat read
    - min_intron_unspliced: minimum intron ratio in any read exon to call it unspliced
    '''
    # exon bed
    if exon_intersect is not None and not confident_gene:
        exon_bed = read_intersect_bed(exon_intersect)
        exon_bed = exon_bed[exon_bed['aln_overlap'] >= min_exon_on_read]
        exon_grouped = exon_bed.groupby('read')
        for read_id, exon_bed in exon_grouped:
            read = reads[read_id]
            exon_bed = exon_bed[exon_bed['overlap']==exon_bed['overlap'].max()]
            gene_id = exon_bed['rep_location'].iloc[0]
            read.gene['id'] = gene_id
            read.gene['name'] = geneid_to_name[gene_id]
            # check the strand
            if paired_end:
                # NOTE: paired end reads are not stranded, single end reads are
                # cannot detect antisense txs in cellranger's way
                read.strand = exon_bed['rstrand'].iloc[0]
                read.feature = 'gene'
            else:
                exon_bed = exon_bed[exon_bed['rstrand']==read.strand]
                if exon_bed.empty:
                    read.feature = 'an'
                else:
                    read.feature = 'gene'
            reads[read_id] = read

    # repeat & intron bed
    bed = read_intersect_bed(rep_int_intersect)
    long_repeat_ratio = 1.5
    short_repeat_ratio = 0.5
    # ratios between 0.5-1.5 means read & repeat have similar lengths
    pass_conditions = (
        (bed['rlen_ratio'] >= long_repeat_ratio) & 
        (bed['aln_overlap'] >= min_overlap)
    ) | (
        (bed['rlen_ratio'] <= short_repeat_ratio) & 
        (bed['r_overlap'] >= min_overlap)
    ) | (
        (bed['r_overlap'] >= min_overlap) & 
        (bed['aln_overlap'] >= min_overlap)
        | (
        bed['repeat'] == 'intron'
        )
    )
    bed = bed[pass_conditions].reset_index(drop=True)
    bed['rfam'] = bed['repeat'].str.split('|').str.get(1)
    bed['overlap_start'] = bed[['start', 'rstart']].max(axis=1)
    bed['overlap_end'] = bed[['end', 'rend']].min(axis=1)

    bed_grouped = bed.groupby('read')
    for read_id, read_bed in bed_grouped:
        read_bed = read_bed.drop_duplicates()
        repeat_bed = read_bed[read_bed['repeat'] != 'intron']
        intron_bed = read_bed[(read_bed['repeat'] == 'intron') & 
                            (read_bed['gene'] == read_bed['rep_location'])]
        
        read = reads[read_id]
        if read.feature != 'gene':
            # check if it is a repeat or intronic read
            location = strand = feature_type = repeat_fam = roverlaps = '.'
            if not repeat_bed.empty:
                repeat_prop = determine_repeat_prop(repeat_bed)
                if repeat_prop >= min_repeat_on_read:
                    feature_type = categorize_repeat(repeat_bed)

                    repeat_bed = repeat_bed.sort_values(by='overlap', ascending=False)
                    repeat_fam0 = repeat_bed['repeat'].to_list()
                    repeat_strand = repeat_bed['rstrand'].to_list()
                    repeat_fam = [f'{f}|{s}' for f, s in zip(repeat_fam0, repeat_strand)]
                    roverlaps = repeat_bed['overlap'].to_list()

                    non_dot_bed = repeat_bed[repeat_bed['rep_location'] != '.']
                    location = non_dot_bed['rep_location'].iloc[0] if not non_dot_bed.empty else '.'
            
            if feature_type == '.' and not intron_bed.empty:
                intron_bp = (intron_bed['overlap_end'] - intron_bed['overlap_start']).sum()
                intron_prop = intron_bp / intron_bed.iloc[0]['read_len']
                if intron_prop >= min_intron_on_read:
                    feature_type = 'intron'
                    non_dot_bed = intron_bed[intron_bed['rep_location'] != '.'].sort_values(by='overlap', ascending=False)
                    location = non_dot_bed['rep_location'].iloc[0] if not non_dot_bed.empty else '.'

            if feature_type not in ['intron', '.']:
                unique_strands = repeat_bed['rstrand'].unique()
                if len(unique_strands) == 1:
                    strand = unique_strands[0]
                    
            new_read = update_nongenics(
                confident_gene, read, geneid_to_name, strand=strand, 
                location=location, feature_type=feature_type, 
                rfam=repeat_fam, roverlaps=roverlaps
            )
            reads[read_id] = new_read
        else:
            # check if it is unspliced for all genic reads given that we have intron intersections
            reads[read_id] = check_genic_unspliced(read, intron_bed, min_intron_unspliced)
        
        if read_info is not None:
            this_read_info = reads[read_id].write_read_info()
            read_info.write(this_read_info.encode())
    return reads

def read_intersect_bed(bed_path):
    bed_header = ['chrom','start','end','exon','exon_len','read_len','read', 'gene', 'tx',
                    'rchrom','rstart','rend','repeat','rstrand','rep_location','overlap']
    bed = pd.read_csv(bed_path, sep="\t", header=None, names=bed_header)
    bed['rlen_ratio'] = (bed['rend'] - bed['rstart']) / bed['exon_len'] # repeat_len/exon_len
    bed['aln_overlap'] = bed['overlap'] / bed['exon_len']
    bed['r_overlap'] = bed['overlap'] / (bed['rend'] - bed['rstart']) # repeat_len/repeat_len
    return bed

def determine_repeat_prop(repeat_bed):
    # merge overlap coordinates and sum up the length
    coords = list(zip(repeat_bed['overlap_start'], repeat_bed['overlap_end']))
    coords.sort()
    merged = []
    start, end = coords[0]
    for s, e in coords[1:]:
        if s <= end: 
            end = max(end, e)
        else:
            merged.append((start, end))
            start, end = s, e
    merged.append((start, end))
    merged_len = sum(e-s for s, e in merged)
    return merged_len / repeat_bed['read_len'].iloc[0]

def categorize_repeat(repeat_bed):
    if (repeat_bed['rfam'] == 'Alu').all():
        return 'Alu'
    elif (repeat_bed['rfam'] == 'Alu').any():
        return 'Alu_oth'
    return 'oth'

def update_nongenics(confident_gene, read, geneid_to_name, **args):
    feature_type = args.get('feature_type')
    if confident_gene:
        # when confident_gene is True, reads passed here are all non genic reads
        # thus update their strand & feature
        read.strand = args.get('strand')
        read.feature = feature_type
    elif feature_type != '.':
        read.feature = feature_type
    
    location = args.get('location')
    if location in geneid_to_name:
        read.gene['id'] = location
        read.gene['name'] = geneid_to_name[location]
        read.repeat['pos'] = 'genic'
    elif location != '.':
        read.repeat['pos'] = location
    
    rfam = args.get('rfam')
    if rfam and rfam != '.':
        read.repeat['fam'] = rfam
        read.repeat['overlap'] = args.get('roverlaps')
    return read

def check_genic_unspliced(read_obj, intron_bed, min_intron_unspliced):
    '''
    Given an exon from a read, check if it's unspliced then put into a 
    category (S - fully spliced, U - unspliced):
      1. Monoexnoic read:
        * overlapped with BOTH exon and intron
      2. Multiexonic read:
        * for middle exon: cover any FULL intron
        * for last exon: intron ratio >= min_intron_unspliced
    '''
    if intron_bed.empty:
        return read_obj
    
    # preset is S
    if read_obj.is_monoexonic:
        if intron_bed['r_overlap'].max() >= min_intron_unspliced:
            read_obj.struct = 'U'
    else:
        gene_strand = read_obj.gene['name'].split('|')[1]
        for exon, exon_bed in intron_bed.groupby('exon'):
            max_overlap = exon_bed['r_overlap'].max()
            # read's exon order is always 5' -> 3'
            if (gene_strand == '+' and exon == 'last_exon') or (gene_strand == '-' and exon == 'exon1'):
                if max_overlap >= min_intron_unspliced:
                    read_obj.struct = 'U'
                    break
            elif max_overlap > 0.99:
                read_obj.struct = 'U'
                break
    return read_obj
    
