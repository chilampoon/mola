"Mappings between mismatches and reads"

from mola.read.process_alignment import *
from mola.mutation.mismatch_site import *
from mola.parse.parse_gff import gff2bed, fetch_gene_names
from mola.parse.parse_bed import site_anno_script

# minimal minor alleles:
# pacbio: 1
# ont: 2
# illumina: 2 or 1?


def map_read_mismatch(
        bam_path,
        pileup_vcf_path,
        ref_vcf_path,
        stranded,
        paired_end,
        min_depth,
        min_minor_allele,
        startswith_chr,
        num_threads,
        reads_dir,
        out_dir,
        tmp_dir
    ):
    if not futils.is_command_avail('bedtools'):
        logging.error(
            'Make sure bedtools is installed, try "conda install -c bioconda bedtools"'
        )
        sys.exit(1)
    if platform.system() == 'Darwin' and not (futils.is_command_avail('gawk') or futils.is_command_avail('gcut')):
        logging.error(
            'Make sure gawk/gcut is installed, try "brew install gawk/gcut"'
        )
        sys.exit(1)
    futils.startswith_chr = startswith_chr

    logging.info('Start mapping mismatches to reads...')
    mismatch_blocks = extract_mismatch_blocks(pileup_vcf_path, min_minor_allele)
    chrom_list = futils.sort_chroms(list(mismatch_blocks.keys()))
    
    sites_dir = os.path.join(os.path.expanduser(out_dir), 'objects', 'sites')
    os.makedirs(sites_dir, exist_ok=True)
    tmp_dir = os.path.abspath(os.path.expanduser(tmp_dir))
    futils.write_base_script(
        script=site_anno_script, 
        out=f'{tmp_dir}/intersect_sites.sh'
    )
    
    # separate gff3 & repeat bed by chromosomes
    site_table_dir = f'{out_dir}/tables'
    os.makedirs(site_table_dir, exist_ok=True)

    strd_out = f'{site_table_dir}/sites_stranded.tsv.gz'
    s_header = SITE_HEADERS['stranded_site_table']
    with futils.write_text(strd_out) as s:
        s.write(futils.list2line(s_header).encode())
    
    if not stranded:
        unstrd_out = f'{site_table_dir}/sites_unstranded.tsv.gz'
        us_header = SITE_HEADERS['unstranded_site_table']
        with futils.write_text(unstrd_out) as us:
            us.write(futils.list2line(us_header).encode())

    _ = futils.process_chrom_in_threads(
        process_chromosome,
        chrom_list,
        num_threads,
        startswith_chr=startswith_chr,
        bam_path=bam_path,
        mismatch_blocks=mismatch_blocks,
        ref_vcf_path=ref_vcf_path,
        stranded=stranded,
        paired_end=paired_end,
        min_depth=min_depth,
        min_minor_allele=min_minor_allele,
        reads_dir=reads_dir,
        sites_dir=sites_dir,
        tmp_dir=tmp_dir
    )
    strd_files = sorted(
        glob.glob(f'{tmp_dir}/*.sites_stranded.tsv.gz'), 
        key=futils.sort_chrom_files
    )
    futils.concat_files(strd_files, strd_out)

    if not stranded:
        unstrd_files = sorted(
            glob.glob(f'{tmp_dir}/*.sites_unstranded.tsv.gz'), 
            key=futils.sort_chrom_files
        )
        futils.concat_files(unstrd_files, unstrd_out)

def process_chromosome(chrom, startswith_chr, bam_path, mismatch_blocks, ref_vcf_path, stranded,
                paired_end, min_depth, min_minor_allele, reads_dir, sites_dir, tmp_dir):
    logging.info(chrom)
    reads = futils.load_gz_pickle(f'{reads_dir}/{chrom}_reads.pkl.gz')
    
    out_bed_path = f'{tmp_dir}/{chrom}.site.bed'
    if ref_vcf_path is not None:
        ref_vcf = pysam.VariantFile(ref_vcf_path)
    else:
        ref_vcf = None
    
    with pysam.AlignmentFile(bam_path) as bam:
        sites, reads = match_mm_to_read(
            chrom, startswith_chr, bam, reads, mismatch_blocks, ref_vcf,
            stranded, paired_end, min_depth, min_minor_allele, out_bed_path
        )
    if ref_vcf:
        ref_vcf.close()
    
    # intersect site with repeat/gene beds
    #gff2bed(f'{tmp_dir}/{chrom}.gff3', f'{tmp_dir}/{chrom}.gff3.bed') it's done in annotate_reads
    exec_intersect(tmp_dir, chrom)
    # add intersections to Sites
    sites = add_anno_info(sites, tmp_dir, chrom)

    gff3_path = f'{tmp_dir}/{chrom}.gff3'
    geneid_to_name = fetch_gene_names(gff3_path)
    sites = write_site_table(geneid_to_name, stranded, chrom, sites, tmp_dir) # sites also got updated

    futils.save_gz_pickle(f'{reads_dir}/{chrom}_reads.pkl.gz', reads)
    futils.save_gz_pickle(f'{sites_dir}/{chrom}_sites.pkl.gz', sites)

    logging.info(f'{chrom} done, {len(sites)} sites')
    
def extract_mismatch_blocks(pileup_vcf_path, min_minor_allele):
    '''
    collect mismatch locations and nucleotides,
    return mismatch blocks.
    NOTE: vcf is 1-base. e.g.
    chr1    3630806 .       T       C
    VCF NEEDS TO BE SORTED
    output is 0-base
    '''
    # assume the longest read is 30000 bp
    block_len = 130000
    mm_blocks = defaultdict(dict) # by chromosomes
    chrom = None
    block_start = 0
    last_pos = -1
    mm_list = []

    with futils.open_text(pileup_vcf_path) as vcf:
        for line in vcf:
            r = futils.read_line(line, sep='\t')
            if not r[0].startswith('#'):
                break
        
        for line in itertools.chain([line], vcf):
            r = futils.read_line(line, sep='\t')
            var_info = r[7]
            if kickout_mismatch(var_info, min_minor_allele):
                continue

            pos = int(r[1]) - 1
            ref_ale, alt_ale = r[3:5]
            if ref_ale == 'N' or alt_ale == 'N':
                continue
            new_chrom = r[0]

            if new_chrom != chrom or pos > block_start + block_len:
                # reset
                if mm_list:
                    # add the last mm_list for last chr
                    add_mm_list(block_start, last_pos+1, chrom, mm_blocks, mm_list)
                    mm_list = []
                block_start = pos
                chrom = new_chrom

            # build blocks
            mm = (pos, ref_ale, alt_ale)
            mm_list.append(mm)
            last_pos = pos

        # add the last one
        if mm_list:
            add_mm_list(block_start, last_pos+1, chrom, mm_blocks, mm_list)
    
    return mm_blocks

def kickout_mismatch(var_info, min_minor_allele):
    '''Only for cellsnp vcf for now'''
    try:
        info = var_info.split(';')
        minor_allele_cnt = [int(i.split('=')[1]) for i in info if i.split('=')[0]=='AD'][0]
        return True if minor_allele_cnt < min_minor_allele else False
    except:
        return False

def add_mm_list(start, end, chrom, mm_blocks, mm_list):
    in_chrom = futils.check_chr_id(chrom)
    mm_blocks[in_chrom][(start, end)] = mm_list

def match_mm_to_read(chrom, startswith_chr, bam, reads, mismatch_blocks, ref_vcf, 
                stranded, paired_end, min_depth, min_minor_allele, out_bed_path):
    '''in one chromosome'''
    mm2read = defaultdict(dict)

    # collect reads and their matching bases for each mismatch
    mm_blocks_chrom = mismatch_blocks[chrom]
    for (s, e) in mm_blocks_chrom:
        mismatches = mm_blocks_chrom[(s, e)]
        alignments = bam.fetch(chrom, s, e)

        # what if some reads overlapping between blocks - just print out all of them
        for aln in alignments:
            read_id = aln.query_name
            if read_id not in reads:
                if paired_end and f'{read_id}/2' in reads:
                    # recover from the processing of paired-end reads
                    read_id = f'{read_id}/2'
                else:
                    continue
            aln_read = reads[read_id]
            if not Read.same_aln(aln_read, aln):
                continue
            aln_ref_pos = set(aln.get_reference_positions()) # 0-base
            for mm_info in mismatches:
                # check if mismatch position is on read first
                if mm_info[0] not in aln_ref_pos:
                    continue
                find_mismatch(mm_info, aln, mm2read)

    # filter mismatches and update Reads
    sites = {}
    with futils.write_text(out_bed_path) as out:
        for mm_info, reads_base in mm2read.items():
            if len(reads_base) < min_depth:
                continue
            mm_pos, major_allele, minor_allele = mm_info
            # initialize object
            site = MismatchSite(
                chrom = chrom,
                pos = mm_pos, # 1-base already
                mismatch = f'{major_allele}>{minor_allele}',
                total = len(reads_base)
            )

            read_feature_types = defaultdict(int)
            for read_id, base in reads_base.items():
                read = reads[read_id]
                read.mismatch[mm_pos] = base
                strand = read.get_strand(genic=True, stranded=stranded) # made sure strand is from gene annotation
                site.base_cnts[base] += 1
                
                read_feature_types[read.feature] += 1
                if read.is_monoexonic:
                    site.read_spliced['U'] += 1
                else:
                    site.read_spliced['S'] += 1

                site.reads.append(f'{read_id}|{strand}|{base}')
                site.cells.append(read.cb)
                
            _, minor_cnt, _, _ = site.major_minor_stats
            if minor_cnt < min_minor_allele:
                continue
            
            # update site info
            site.mismatch = site.mismatch_from_cnt() # could be different from the original mismatch
            site.read_comp = dict(read_feature_types)
            site.snp = get_snp_type(startswith_chr, ref_vcf, chrom, mm_pos, major_allele, minor_allele)
            ## output bed for intersecting with repeat/gene annotation
            out.write(site.write_bed())
            sites[(chrom, site.pos)] = site
    
    return sites, reads

def find_mismatch(mm_info, aln, mm2read):
    # mm_info ex: (170019576, 'T', 'C')
    mm_pos, allele1, allele2 = mm_info
    read_positions = aln.get_reference_positions(full_length=True)
    # quick check the length
    if len(read_positions) != aln.query_length:
        raise('Need to check ref sequences for ' + aln.query_name)
    
    # grab the actual nucleotide
    read_nt = aln.seq[read_positions.index(mm_pos)]
    mm2read[(mm_pos+1, allele1, allele2)][aln.query_name] = read_nt

def get_snp_type(startswith_chr, ref_vcf, chrom, pos, major_allele, minor_allele):
    if ref_vcf is None:
        return '.'

    try:
        mismatch = ref_vcf.fetch(chrom, pos-1, pos)
    except ValueError:
        # chromosome not in vcf, could be due to w/wo chr prefix
        chrom = chrom.replace('chr', '') if startswith_chr else f'chr{chrom}'
        try:
            mismatch = ref_vcf.fetch(chrom, pos-1, pos)
        except ValueError:
            # chromosome really not in vcf
            return 'none'
    
    mm_alleles = set([major_allele, minor_allele])
    for entry in mismatch:
        if entry.pos == pos:
            if len(set(entry.alleles).intersection(mm_alleles)) != len(mm_alleles):
                continue
            if 'COMMON' in entry.info and entry.info['COMMON'] == 1:
                return 'common'
            else:
                return 'rare'
    return 'none'

def exec_intersect(tmp_dir, chrom):
    # run bedtools
    site_bed = f'{tmp_dir}/{chrom}.site.bed'
    rep_intron_bed = f'{tmp_dir}/{chrom}.repeat.intron.bed'
    gff3_bed = f'{tmp_dir}/{chrom}.gff3.bed'
    bash_path = f'{tmp_dir}/intersect_sites.sh'
    cmd = ['bash', bash_path, site_bed, rep_intron_bed, gff3_bed, tmp_dir, chrom]
    subprocess.run(cmd, check=True)

def add_anno_info(sites, tmp_dir, chrom):
    rep_intron_intersect = f'{tmp_dir}/{chrom}.site.repeat.intron.bed'
    gff3_intersect = f'{tmp_dir}/{chrom}.site.gff3.bed'

    with futils.open_text(rep_intron_intersect) as rep_bed:
        for line in rep_bed:
            l = futils.read_line(line, sep='\t')
            chrom, start, end, element, strand, feature_id = l # start = pos-1
            if element == 'intron':
                region = f'{element}@{feature_id}'
                sites = _add_gene_reg(sites, chrom, end, region)
            else:
                region = f'{element}|{strand}'
                sites[(chrom, int(end))].repeat.add(region)
    
    with futils.open_text(gff3_intersect) as gff_bed:
        for line in gff_bed:
            l = futils.read_line(line, sep='\t')
            chrom, start, end, region, strand, gene_id = l
            region = f'{region}@{gene_id}'
            sites = _add_gene_reg(sites, chrom, end, region)
    return sites

def _add_gene_reg(sites, chrom, end, region):
    if 'reg' not in sites[(chrom, int(end))].gene:
        sites[(chrom, int(end))].gene['reg'] = set([region])
    else:
        sites[(chrom, int(end))].gene['reg'].add(region)
    return sites

def write_site_table(geneid_to_name, stranded, chrom, sites_obj, out_dir):
    '''pseudobulk sites
    Site table inputs for inference:
      - 1. sites with strand, i.e. from reads mapped well to genes
      - 2. sites without strand, i.e. all sites with enough coverage
    
    Returns:
        Sites with newly added stats
    '''
    strd_out = f'{out_dir}/{chrom}.sites_stranded.tsv.gz'
    unstrd_out = f'{out_dir}/{chrom}.sites_unstranded.tsv.gz'

    with futils.write_text(strd_out) as s:
        if not stranded:
            us = futils.write_text(unstrd_out)

        for site in sites_obj.values():
            # add gene name & strand
            if 'reg' in site.gene:
                gene_ids = set([r.split('@')[1] for r in site.gene['reg']])
                gene_names = set([geneid_to_name[g] for g in gene_ids])
                gene_names = ';'.join(sorted(list(gene_names)))
                site.gene['id'] = ';'.join(gene_ids)
                site.gene['name'] = gene_names
            
            if not stranded:
                # includes stranded & unstranded sites
                us.write(site.write_table(stranded=False).encode())
            if site.is_from_genic or stranded:
                s.write(site.write_table(stranded=True).encode())
    
    if not stranded: us.close()
    return sites_obj
