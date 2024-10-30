from mola.read.process_alignment import *

######## 10x 3prime v3 kit ########
# 5' -> 3'
# Read1-barcode-umi-polyT-cDNA-TSO
# -22bp--16bp---12bp-30bp------30bp
# 
# others to be added
###################################

LIB_STRUCT = {
    '10x_3prime_v3': {
        'read1': 'CTACACGACGCTCTTCCGATCT',
        'tso': 'CCCATGTACTCTGCGTTGATACCACTGCTT',
        'rev_tso': 'AAGCAGTGGTATCAACGCAGAGTACATGGG',
        'polyA': 'A' * 7,
        'polyT': 'T' * 7
    }
}


# for long read: trim in a 3'->5' way (with polyA tail)
# check first 35bp (tso) & last 110bp (polyA+others)
SETTINGS = {
    'long': {
        'trim_head': 35,
        'trim_tail': 110,
        'max_ed_dist': 2
    }, 
    'short': {
        'trim_head': 35,
        'trim_tail': 35,
        'max_ed_dist': 1
    }
}


def sliding_window(seq, target_seq, k, max_ed_dist, head=True):
    trim_pos = None
    kmers = get_kmers(target_seq, k=k)
    for idx, kmer in enumerate(kmers):
        if edit_distance(seq, kmer) <= max_ed_dist:
            if head:
                trim_pos = idx + k
            else:
                trim_pos = idx
            break
    return trim_pos

TAGS = ['CB', 'CR', 'UB', 'UR', 'XM']
def write_fq_with_cb_umi(aln, tags_include=TAGS):
    try:
        read_qual = ''.join(map(lambda x: chr(x+33), aln.query_qualities))
        all_tags = aln.get_tags(with_value_type=True)
        tags_info = '\t'.join([f'{t[0]}:{t[2]}:{t[1]}' for t in all_tags if t[0] in tags_include])
        read_id = '@' + aln.query_name + '\t' + tags_info
        return f'{read_id}\n{aln.seq}\n+\n{read_qual}\n'
    except Exception as e:
        logging.debug(f'{e} from aln:\n{aln.query_name}')
        return None

def trim_long_aln(aln, trim_head, trim_tail, tso, rev_tso, polyA, polyT, max_ed_dist):
    '''
    trim sequences polyA & after polyA + TSO,
    same as the rev complemented strand.
    Utlize polyA &/or uncorrected barcode tag from BAM
    '''
    cr = [i[1] for i in aln.tags if i[0]=='CR'][0]
    cr = rev_complement(cr)
    ## make sure bam is tagged with uncorrected barcode, warnings?##

    # polyT head (5'->3') turns to polyA tail
    if not aln.is_reverse:
        seq = rev_complement(aln.seq)
    else:
        seq = aln.seq

    # locate cutting position in tail
    polyA_pos = seq[-trim_tail:].find(polyA)
    if polyA_pos != -1:
        trim_pos_tail = len(seq) - (trim_tail - polyA_pos)
        if seq[-trim_tail:].find(cr) == -1:
            print(f'check {aln.query_name}, CR: {cr}\n {seq}')
    else:
        # if cannot find polyA, just use cr's position
        cr_pos = seq[-trim_tail:].find(cr)
        if cr_pos >= 55:
            trim_pos_tail = len(seq) - (trim_tail - cr_pos)
            # try to search for polyA with insertions (max=2)
            attempt_sizes = [7, 8, 9]
            max_err = 2
            attempt_check = None
            for s in attempt_sizes:
                #10 is the umi size 12-2
                attempt = seq[trim_pos_tail-s-10:trim_pos_tail-10]
                if attempt.count('A') >= s - max_err:
                    attempt_check = s+10
            if attempt_check:
                trim_pos_tail -= attempt_check
        else:
            # no match, not trim
            trim_pos_tail = len(seq)

    # locate cutting position in head
    ## using partial tso (last 20bp)
    partial_size = 16
    trim_pos_head = None
    tso_pos = seq[:trim_head].find(rev_tso[-partial_size:])
    if tso_pos != -1:
        trim_pos_head = tso_pos + partial_size
    else:
        # try to find any fuzzy match
        ## use levenshtein here in case of indels, slower :/
        throw = len(rev_tso) - partial_size - 3 # 3 is a buffer
        trim_pos_head = sliding_window(
            rev_tso[-partial_size:], 
            seq[throw:trim_head],
            k=partial_size,
            max_ed_dist=max_ed_dist,
            head=True
        )
        if trim_pos_head is None:
            # no match, not trim
            trim_pos_head = 0

    # now to deal with reads with multiple molecules sticked together
    ## just pick the last one as sockeye only extracts the last barcode..
    ## TODO - update barcode umi for partial artifact (& if the seq is accurate)##
    new_seq = seq[trim_pos_head:trim_pos_tail]
    new_quals = aln.query_qualities[trim_pos_head:trim_pos_tail]
    extra_polyA_pos = [m.end() for m in re.finditer(polyA, new_seq)]
    if extra_polyA_pos:
        last_polyA_end = max(extra_polyA_pos)
        # no fuzzy searching here
        extra_tso_pos = new_seq[last_polyA_end:].find(rev_tso[-partial_size:])
        if extra_tso_pos != -1:
            trim_pos_head = last_polyA_end + extra_tso_pos + partial_size
            new_seq = new_seq[trim_pos_head:]
            new_quals = new_quals[trim_pos_head:]

    # final check
    if trim_pos_tail != len(seq):
        fc_window = 30 # outside of trim_tail
        new_a = new_seq[-fc_window:].find('A'*6)
        if new_a != -1:
            new_seq = new_seq[:-(fc_window-new_a)]
            new_quals = new_quals[:-(fc_window-new_a)]

    # replace old info
    if not aln.is_reverse:
        aln.seq = rev_complement(new_seq)
    else:
        aln.seq = new_seq
    aln.query_qualities = new_quals
    fq_lines = write_fq_with_cb_umi(aln)

    return fq_lines

def trim_short_aln(aln, trim_head, trim_tail, tso, rev_tso, polyA, polyT, max_ed_dist):
    '''
    Note that here is just focusing on trimming TSO/polyA/T in read 2 from 10x data
    Ideal input: bam from cellranger

    not using aligner's strand here
        * forward: polyT in the head, tso in the tail
        * reverse: rev_tso in the head, polyA in the tail

    Trim polyA/T in a harcoded way not (len=10)
    '''
    partial_size = 13
    trim_pos_head = None
    trim_pos_tail = None
    seq = aln.seq

    # checking the head
    rtso_pos = seq[:trim_head].find(rev_tso[-partial_size:]) #last couple bps
    if rtso_pos != -1:
        trim_pos_head = rtso_pos + partial_size
    else:
        trim_pos_head = sliding_window(
            rev_tso[-partial_size:], 
            seq[:trim_head],
            k=partial_size,
            max_ed_dist=max_ed_dist,
            head=True
        )

    # checking the tail
    tso_pos = seq[-trim_tail:].find(tso[:partial_size])
    if tso_pos != -1:
        trim_pos_tail = len(seq) - trim_tail + tso_pos
    else:
        trim_pos_tail = sliding_window(
            tso[:partial_size],
            seq[-trim_tail:],
            k=partial_size,
            max_ed_dist=max_ed_dist,
            head=False
        )
        if trim_pos_tail is not None: 
            trim_pos_tail += len(seq) - trim_tail

    # checking polyA/T
    if trim_pos_head is None and trim_pos_tail is None:
        if seq[:len(polyT)] == polyT:
            trim_pos_head = len(polyT)
        else:
            # no match, not trim
            trim_pos_head = 0

        if trim_pos_head > 0:
            # has trimmed the head, then don't trim tail
            trim_pos_tail = len(seq)
        elif seq[-len(polyA):] == polyA:
            trim_pos_tail = len(seq) - len(polyA)
        else:
            # no match, not trim
            trim_pos_tail = len(seq)
    if trim_pos_head is None: trim_pos_head = 0
    if trim_pos_tail is None: trim_pos_tail = len(seq)

    # trim
    quals = aln.query_qualities
    aln.seq = seq[trim_pos_head:trim_pos_tail]
    aln.query_qualities = quals[trim_pos_head:trim_pos_tail]
    fq_lines = write_fq_with_cb_umi(aln)

    return fq_lines

def has_clippings(aln):
    cigar = aln.cigarstring
    if 'S' not in cigar:
        return False
    return True

def trim_reads(bam, out_dir, library, max_edit_dist, short):
    assert library in LIB_STRUCT.keys(), f'Library {library} not supported'
    lib_structure = LIB_STRUCT[library]
    tso = lib_structure.get('tso')
    rev_tso = lib_structure.get('rev_tso')
    polyA = lib_structure.get('polyA')
    polyT = lib_structure.get('polyT')
    
    if not short:
        settings = SETTINGS['long']
        trim_func = trim_long_aln
    else:
        settings = SETTINGS['short']
        trim_func = trim_short_aln
    trim_head = settings.get('trim_head')
    trim_tail = settings.get('trim_tail')
    if max_edit_dist is None:
        max_edit_dist = settings.get('max_ed_dist')

    out_fq = os.path.join(out_dir, f'{".".join(os.path.basename(bam).split(".")[:-1])}.trimmed.fastq.gz')

    logging.info('Trimming reads, takes a while...')
    with pysam.AlignmentFile(bam, 'rb') as bam_in:
        with futils.write_text(out_fq) as fq_out:
            for aln in bam_in:
                if not has_clippings(aln):
                    fq_lines = write_fq_with_cb_umi(aln)
                    fq_out.write(fq_lines.encode())
                    continue
                    
                trimmed_read = trim_func(aln, trim_head, trim_tail, tso, rev_tso, polyA, polyT, max_edit_dist)
                if trimmed_read is not None:
                    fq_out.write(trimmed_read.encode())
    
    logging.info('Trimming done!')
