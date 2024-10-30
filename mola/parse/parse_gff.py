from mola.parse.file_utils import *

futils = FileUtils()

def fix_gff3_id(gff3, out):
    '''
    fix CHM13v2.0 gff3 issue -
    the problem is ids for some start&stop codons/exons are not unique
    11/20/2022
    '''
    start_codon_cnt = defaultdict(lambda:0)
    stop_codon_cnt = defaultdict(lambda:0)
    liftoff_id_cnt = defaultdict(lambda:0)

    with open(gff3, 'r') as f, gzip.open(out, 'wb') as out:
        for row in f:
            if row.startswith('#'): 
                out.write(row.encode())
                continue

            r = row.rstrip().split('\t')
            if r[2] == 'start_codon':
                tx_id = extract_id(r[8], 'transcript_id')
                info = sep_info(r[8])
                new_r = update_id(r, info, start_codon_cnt[tx_id])
                row = '\t'.join(new_r) + '\n'
                start_codon_cnt[tx_id] += 1
            elif r[2] == 'stop_codon':
                info = sep_info(r[8])
                tx_id = extract_id(r[8], 'transcript_id')
                new_r = update_id(r, info, stop_codon_cnt[tx_id])
                row = '\t'.join(new_r) + '\n'
                stop_codon_cnt[tx_id] += 1

            if r[1] == 'Liftoff' and r[2] not in ['gene', 'transcript']:
                old_id = extract_id(r[8], 'ID')
                info = sep_info(r[8])
                tmp = old_id.split(':')
                if len(tmp) == 2:
                    id_num = 0
                    this_id = old_id
                    repl = False
                elif len(tmp) == 3:
                    id_num = int(tmp[-1])
                    this_id = ':'.join(tmp[:-1])
                    repl = True
                else:
                    print(f'check {old_id}')
                if id_num != liftoff_id_cnt[this_id]:
                    new_r = update_id(r, info, liftoff_id_cnt[this_id], replace=repl)
                    row = '\t'.join(new_r) + '\n'
                liftoff_id_cnt[this_id] += 1
            out.write(row.encode())

def sep_info(info):
    items_tuple = [i.split('=') for i in info.split(';')]
    return items_tuple

def comb_info(items):
    res = [f'{i[0]}={i[1]}' for i in items]
    return ';'.join(res)

def update_id(r, info, count, replace=False):
    (idx, sid) = [(i, j) for i,j in enumerate(info) if j[0] == 'ID'][0]
    if replace:
        tmp = sid[1].split(':')
        new_id = f'{":".join(tmp[:-1])}:{count}'
    else:
        new_id = f'{sid[1]}:{count}'
    info[idx] = (sid[0], new_id)
    r[8] = comb_info(info)
    return r

def gff2bed(gff_path, out_path):
    transcripts = defaultdict(list)

    with futils.open_text(gff_path) as gff:
        for line in gff:
            l = futils.read_line(line)
            if not l[0].startswith('#'):
                break

        for line in itertools.chain([line], gff):
            l = futils.read_line(line)
            if len(l) < 9 or l[2] in ('gene', 'transcript'):
                continue

            gene_id = extract_id(l[8], 'gene_id')
            transcript_id = extract_id(l[8], 'transcript_id')

            if gene_id and transcript_id:
                transcripts[transcript_id].append({
                    'chrom': l[0],
                    'start': l[3],
                    'end': l[4],
                    'type': l[2],
                    'strand': l[6],
                    'gene': gene_id,
                    'tx': transcript_id
                })

    # process the transcripts and rename first & last exons
    with futils.write_text(out_path) as out:
        for tx_records in transcripts.values():
            tx_records.sort(key=lambda x: int(x['start']))

            # handle exon renaming
            strand = tx_records[0]['strand']
            exon_indices = [i for i, record in enumerate(tx_records) if record['type'] == 'exon']

            if len(exon_indices) > 1:
                if strand == '+':
                    first_exon_idx = exon_indices[0]
                    last_exon_idx = exon_indices[-1]
                else:
                    first_exon_idx = exon_indices[-1]
                    last_exon_idx = exon_indices[0]
            else:
                first_exon_idx = last_exon_idx = None

            for i, r in enumerate(tx_records):
                if i == first_exon_idx:
                    reg_type = 'first_exon'
                elif i == last_exon_idx:
                    reg_type = 'last_exon'
                else:
                    reg_type = r['type']
                out_line = [r['chrom'], r['start'], r['end'], reg_type, r['strand'], r['gene'], r['tx']]
                out.write(futils.list2line(out_line))

def build_regex(id_key, pattern):
    return re.compile(pattern.format(id_key=id_key))

def extract_id(attributes, id_key):
    # build the regex patterns
    gff3_pattern = r'(?<![\w-])({id_key})=([^;]+)'
    gtf_pattern = r'(?<![\w-])({id_key}) "([^"]+)"'
    gff3_regex = build_regex(id_key, gff3_pattern)
    gtf_regex = build_regex(id_key, gtf_pattern)
    
    # gff3
    match = re.search(gff3_regex, attributes)
    if match:
        return match.group(2)
    # gtf
    match = re.search(gtf_regex, attributes)
    if match:
        return match.group(2)
    
    return None

def fetch_gene_names(gtf):
    id2name = {}
    with futils.open_text(gtf) as gtf:
        for line in gtf:
            l = futils.read_line(line)
            if l[0].startswith('#') or l[2] != 'gene':
                continue
            strand, attr = l[6], l[8]
            gene_name = extract_id(attr, 'gene_name|gene')
            gene_id = extract_id(attr, 'gene_id')
            gene_name = gene_id if not gene_name else gene_name
            id2name[gene_id] = f'{gene_name}|{strand}'
    return id2name
