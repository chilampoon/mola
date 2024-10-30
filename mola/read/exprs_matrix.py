from mola.parse.file_utils import *
import logging

futils = FileUtils()

# matrix (column) for bulk data
def get_gene_matrix(
        obj_dir,
        paired_end,
        mode,
        concat_prefixes,
        out_dir,
        gene_id,
        num_threads
    ):
    '''
    Count reads on each gene for bulk data.
    '''
    add_prefix, feature_types, out_name = parse_mode(mode, concat_prefixes)
    out_file = os.path.join(out_dir, f'{out_name}.tsv.gz')
    logging.info(f'[bulk] Generating {mode} count table...')

    chrom_list = list(map(
        lambda x: os.path.basename(x).replace('_reads.pkl.gz', ''), 
        glob.glob(f'{obj_dir}/reads/*_reads.pkl.gz')
    ))
    chrom_list = futils.sort_chroms(chrom_list)

    logging.info('[bulk] Start counting reads...')
    chrom_res = futils.process_chrom_in_threads(
        process_chromosome_bulk,
        chrom_list,
        num_threads,
        paired_end=paired_end,
        gene_id=gene_id,
        obj_dir=obj_dir,
        add_prefix=add_prefix,
        feature_types=feature_types
    )

    with futils.write_text(out_file) as fout:
        fout.write('#feature\tcount\n'.encode()) #header
        for chrom in chrom_list:
            chrom_feature2cnts = chrom_res[chrom]
            for feature, cnt in chrom_feature2cnts.items():
                if feature == '.':
                    continue
                fout.write(futils.list2line([feature, cnt]).encode())

def process_chromosome_bulk(chrom, paired_end, gene_id, obj_dir, add_prefix, feature_types):
    logging.info(chrom)
    read_obj_path = os.path.join(obj_dir, f'reads/{chrom}_reads.pkl.gz')
    chrom_feature2cnts = digest_reads_bulk(
        paired_end, read_obj_path, feature_types, add_prefix, gene_id
    )
    return chrom_feature2cnts

def digest_reads_bulk(paired_end, read_obj_path, feature_types, add_prefix, gene_id):
    feature2cnts = defaultdict(int) # {feature:count}
    if paired_end:
        # ensure only one read is counted for each pair
        pe_read_ids = set()
    
    reads = futils.load_gz_pickle(read_obj_path)
    for read in reads.values():
        read_feature = read.get_feature_to_count()
        if read_feature not in feature_types or read.feature == '.':
            continue

        if paired_end: 
            read_id = read.read_id
            if read_id.endswith(('/1', '/2')):
                read_id = re.sub(r"/[12]$", "", read_id)
            if read.read_id in pe_read_ids:
                continue
            else:
                pe_read_ids.add(read.read_id)

        feature_id, feature_name = read.get_feature_id_name()
        fid = feature_id if gene_id else feature_name
        if read.feature in add_prefix:
            fid = f'{add_prefix[read.feature]}.{fid}'
        feature2cnts[fid] += 1
    return feature2cnts


# matrix for single cell data
def get_cell_by_gene_matrix(
        obj_dir,
        splice,
        mode,
        concat_prefixes,
        out_dir,
        gene_id,
        barcode_list,
        num_threads
    ):
    '''
    Generate feature by cell matrix based on read features from read annotation.
    NOTE that no isoform matrix for now.
    '''
    # spliced and unspliced reads are from genic reads
    if splice:
        mode = 'gene' # fixed
        add_prefix, feature_types, out_name = parse_mode(mode, concat_prefixes)
        out_file = [f'{out_dir}/spliced_counts.tsv.gz', f'{out_dir}/unspliced_counts.tsv.gz']
        logging.info(f'[sc] Generating spliced and unspliced count matrices...')
    else:
        add_prefix, feature_types, out_name = parse_mode(mode, concat_prefixes)
        out_file = os.path.join(out_dir, f'{out_name}.tsv.gz')
        logging.info(f'[sc] Generating {mode} matrix...')

    chrom_list = list(map(
        lambda x: os.path.basename(x).replace('_reads.pkl.gz', ''), 
        glob.glob(f'{obj_dir}/reads/*_reads.pkl.gz')
    ))
    chrom_list = futils.sort_chroms(chrom_list)

    barcodes = None
    if barcode_list is not None:
        barcodes = get_barcodes(barcode_list)

    logging.info('[sc] Start counting reads...')
    chrom_res = futils.process_chrom_in_threads(
        process_chromosome_sc,
        chrom_list,
        num_threads,
        gene_id=gene_id, 
        obj_dir=obj_dir,
        add_prefix=add_prefix, 
        feature_types=feature_types,
        splice=splice,
        barcodes=barcodes
    )
    all_features, cell2feature = aggregate_results(chrom_list, chrom_res, splice)
    generate_matrix(all_features, cell2feature, splice, out_file)
    logging.info(f'Done! output to {out_file}')

# add more features here if needed
MODE_TO_FEATURE_TYPES = {
    'gene': ['gene'],
    'Alu': ['Alu_genic', 'Alu_oth_genic'],
    'intron': ['intron'],
    'oth': ['oth_genic'],
    'intergenic': ['intergenic', 'Alu_intergenic', 'Alu_oth_intergenic', 'oth_intergenic'],
    'an': ['an'] # antisense
}
_MODES = list(MODE_TO_FEATURE_TYPES.keys())

def parse_mode(mode, concat_prefixes):
    # (+) = oplus = concat
    ## logic is like (A+B) (+) (C+D)
    add_prefix = {} # {feature_type:prefix}, add prefix to feature names for concat (won't sum)
    operations = set()
    if '(+)' in mode:
        operations.add('concat')
        concat_prefixes = concat_prefixes.split(',')

        mode_cat = mode.split('(+)')
        assert len(concat_prefixes) == len(mode_cat), (
            f'Disconcordant number of concat_prefixes ({len(concat_prefixes)})'
            f' and concat modes ({len(mode_cat)})'
        )
        new_mode = []
        for mode_sum, prefix in zip(mode_cat, concat_prefixes):
            if '+' in mode_sum:
                operations.add('sum')
            for m in mode_sum.split('+'):
                new_mode.append(m)
                for ftype in MODE_TO_FEATURE_TYPES[m]:
                    add_prefix[ftype] = re.sub('\.$', '', prefix)
        mode = new_mode
    elif '+' in mode:
        operations.add('sum')
        mode = mode.split('+')
    else:
        mode = [mode]

    if not set(mode).issubset(_MODES):
        raise ValueError(f'Invalid mode in {mode}')
    op_out = f'{"_".join(sorted(operations))}.' if len(operations) > 0 else ''
    out_name = f'{"_".join(mode)}.{op_out}counts'
    feature_types = set(item for m in mode for item in MODE_TO_FEATURE_TYPES[m])
    return add_prefix, feature_types, out_name

def get_barcodes(barcode_list_path):
    barcodes = set()
    with futils.open_text(barcode_list_path) as fin:
        for line in fin:
            l = line.rstrip().split('\t')
            barcodes.add(l[0])
    return barcodes

def process_chromosome_sc(chrom, gene_id, obj_dir, add_prefix, feature_types, splice, barcodes):
    logging.info(chrom)
    read_obj_path = os.path.join(obj_dir, f'reads/{chrom}_reads.pkl.gz')
    if splice:
        chrom_features, chrom_cell2spliced, chrom_cell2unspliced = digest_reads_by_struct(
            read_obj_path, gene_id, barcodes
        )
        return (chrom_features, (chrom_cell2spliced, chrom_cell2unspliced))
    else:
        chrom_features, chrom_cell2feature = digest_reads_by_feature(
            read_obj_path, feature_types, add_prefix, gene_id, barcodes
        )
        return (chrom_features, (chrom_cell2feature))

def digest_reads_by_feature(read_obj_path, feature_types, add_prefix, gene_id, barcodes):
    '''reads are classified by feature - gene, Alu, intron, etc.
    '''
    cell2feature = defaultdict(lambda: defaultdict(int)) # {barcode:{feature:count}}
    all_features = set()

    reads = futils.load_gz_pickle(read_obj_path)
    for read in reads.values():
        if read.cb == '.' or (barcodes is not None and read.cb not in barcodes):
            continue
        read_feature = read.get_feature_to_count()
        if read_feature not in feature_types or read.feature == '.':
            continue
        
        feature_id, feature_name = read.get_feature_id_name()
        fid = feature_id if gene_id else feature_name
        if read.feature in add_prefix:
            fid = f'{add_prefix[read.feature]}.{fid}'
        cell2feature[read.cb][fid] += 1
        all_features.add(fid)
    return all_features, cell2feature

def digest_reads_by_struct(read_obj_path, gene_id, barcodes):
    '''reads are classified by structure - spliced or unspliced; all genic reads
    '''
    cell2spliced = defaultdict(lambda: defaultdict(int)) # {barcode:{feature:count}}
    cell2unspliced = defaultdict(lambda: defaultdict(int))
    all_features = set()

    reads = futils.load_gz_pickle(read_obj_path)
    for read in reads.values():
        if read.feature != 'gene':
            continue
        if read.cb == '.' or (barcodes is not None and read.cb not in barcodes):
            continue
        read_feature = read.get_feature_to_count()
        
        feature_id, feature_name = read.get_feature_id_name()
        fid = feature_id if gene_id else feature_name
        if read.struct == 'S':
            cell2spliced[read.cb][fid] += 1
        elif read.struct == 'U':
            cell2unspliced[read.cb][fid] += 1
        all_features.add(fid)
    return all_features, cell2spliced, cell2unspliced

def aggregate_results(chrom_list, chrom_res, splice):
    # merge results in this way to prevent duplicated feature names
    all_features = set()
    if splice:
        cell2spliced = defaultdict(lambda: defaultdict(int))
        cell2unspliced = defaultdict(lambda: defaultdict(int))
        for chrom in chrom_list:
            (chrom_features, (chrom_cell2spliced, chrom_cell2unspliced)) = chrom_res[chrom]
            all_barcodes = set(chrom_cell2spliced.keys()) & set(chrom_cell2unspliced.keys())
            for cb in all_barcodes:
                for ft in chrom_features:
                    cell2spliced[cb][ft] += chrom_cell2spliced[cb][ft]
                    cell2unspliced[cb][ft] += chrom_cell2unspliced[cb][ft]
            all_features.update(chrom_features)
            del chrom_res[chrom]
        return all_features, (cell2spliced, cell2unspliced)
    else:
        cell2feature = defaultdict(lambda: defaultdict(int))
        for chrom in chrom_list:
            (chrom_features, chrom_cell2feature) = chrom_res[chrom]
            for barcode, features in chrom_cell2feature.items():
                for fid, cnt in features.items():
                    cell2feature[barcode][fid] += cnt
            all_features.update(chrom_features)
            del chrom_res[chrom]
        return all_features, cell2feature

def generate_matrix(all_features, cell2feature, splice, out):
    all_features = sorted(all_features) # for gene_ids these should all be unique
    # features are unique
    if splice:
        cell2spliced, cell2unspliced = cell2feature
        barcodes = sorted(set(cell2spliced.keys()) & set(cell2unspliced.keys()))
        with futils.write_text(out[0]) as sout, futils.write_text(out[1]) as uout:
            sout.write(futils.list2line(['#feature'] + barcodes).encode())
            uout.write(futils.list2line(['#feature'] + barcodes).encode())
            for feature in all_features:
                if feature == '.':
                    continue
                spliced_cnt = np.array([cell2spliced[cell].get(feature, 0) for cell in barcodes])
                spliced_cnt = [feature] + list(spliced_cnt)
                sout.write(futils.list2line(spliced_cnt).encode())

                unspliced_cnt = np.array([cell2unspliced[cell].get(feature, 0) for cell in barcodes])
                unspliced_cnt = [feature] + list(unspliced_cnt)
                uout.write(futils.list2line(unspliced_cnt).encode())
    else:
        barcodes = sorted(cell2feature.keys())
        with futils.write_text(out) as fout:
            fout.write(futils.list2line(['#feature'] + barcodes).encode()) #header
            for feature in all_features:
                if feature == '.':
                    continue
                feature_cnt = np.array([cell2feature[cell].get(feature, 0) for cell in barcodes])
                feature_cnt = [feature] + list(feature_cnt)
                fout.write(futils.list2line(feature_cnt).encode())


## get haplotype counts for genes
def count_gene_haps(reads_dir, out_dir):
    '''
    Count reads from haplotypes with reads confidently mapped to a gene
    '''
    out = os.path.join(out_dir, 'gene_hap_counts.tsv')
    chrom_list = list(map(
        lambda x: os.path.basename(x).replace('_reads.pkl.gz', ''), 
        glob.glob(f'{reads_dir}/*_reads.pkl.gz')
    ))
    chrom_list = futils.sort_chroms(chrom_list)
    
    with futils.write_text(out) as fout:
        fout.write('chrom\tgene\tloc\ttotal\thap1\thap2\thap_oth\n')
        for chrom in chrom_list:
            read_obj_path = os.path.join(reads_dir, f'{chrom}_reads.pkl.gz')
            reads = futils.load_gz_pickle(read_obj_path)

            # gene coverage {gene:total}
            gene2total = defaultdict(int)
            # gene's haplotype counts {gene:loc:{hap1, hap2}}
            gene2hap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for read_id, read in reads.items():
                if read.feature != 'gene':
                    continue
                gene2total[read.gene.get('name')] += 1
                if read.hap:
                    for h in read.hap:
                        loc_id, read_h = h.split('_')
                        gene2hap[read.gene.get('name')][loc_id][read_h] += 1

            for gene, total in gene2total.items():
                if gene not in gene2hap:
                    loc = '.'
                    hap1 = hap2 = total
                    hap_oth = 0
                    gene = gene.split('|')[0]
                    fout.write(f'{chrom}\t{gene}\t{loc}\t{total}\t{hap1}\t{hap2}\t{hap_oth}\n')
                else:
                    total = gene2total[gene]
                    gene = gene.split('|')[0]
                    for loc, haps in gene2hap[gene].items():
                        hap1, hap2 = haps['1'], haps['2']
                        hap_oth = total - hap1 - hap2
                        fout.write(f'{chrom}\t{gene}\t{loc}\t{total}\t{hap1}\t{hap2}\t{hap_oth}\n')
    