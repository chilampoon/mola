import pysam
import editdistance as ed
from dataclasses import dataclass, field
from mola.parse.file_utils import *


futils = FileUtils()

def aln2Read(aln, primary, min_len, min_mapq):
    # NOTE pos is 1-base
    if kickout_aln(aln, primary, min_len, min_mapq):
        return None
    
    cell_barcode = get_tags(aln, tag_id_list=['CB'])
    cell_barcode = '.' if not cell_barcode else re.sub(r'-[0-9]+', '', cell_barcode) # 10x barcodes mostly
    umi = get_tags(aln, tag_id_list=['XM'])
    if not umi:
        umi = get_tags(aln, tag_id_list=['UB'])
    umi = '.' if not umi else umi
    
    read = Read(
        id = aln.query_name,
        chr = futils.check_chr_id(aln.reference_name),
        start = aln.reference_start + 1,
        end = aln.reference_end + 1,
        len = aln.query_length,
        mapq = aln.mapping_quality,
        cigar = aln.cigarstring,
        cb = cell_barcode,
        umi = umi
    )
    read.struct = 'MO' if read.is_monoexonic else 'MU' # determine if its U later
    return read

def kickout_aln(aln, primary, min_len, min_mapq):
    '''if fails in ANY condition, kick it out'''
    if primary and (aln.is_secondary or aln.is_supplementary):
        return True
    if aln.mapping_quality < min_mapq:
        return True
    if len(aln.query_sequence) < min_len:
        return True
    return False

# CHROMS = [str(i) for i in range(1,23)] + ['X', 'Y', 'MT', 'M']
# CHROMS += [f'chr{c}' for c in CHROMS]

def add_tags(bam_path, tag_path, tag_names, out_dir):
    futils.good_prefix(bam_path, add2last='.tagged')
    out = os.path.join(out_dir, f'{futils.out_prefix}.bam')

    read2tags = read_tag_file(tag_path, tag_names)
    
    logging.info('Writing tagged bam...')
    with pysam.AlignmentFile(bam_path, "rb") as b_in:
        with pysam.AlignmentFile(out, "wb", template=b_in) as b_out:
            for aln in b_in:
                if aln.query_name in read2tags:
                    # replace old tags and add new tags
                    new_tags = read2tags[aln.query_name]
                    old_tags = dict(aln.tags)
                    for tag_id, value in new_tags.items():
                        old_tags[tag_id] = value
                    aln.tags = list(old_tags.items())
                    # if a read doesn't have tags, it's not printed
                    b_out.write(aln)
    pysam.index(out, "-@", "4")

def demux_by_tag(bam_path, tag_demux, out_dir):
    '''
    tags are required
    if input a tag list, then only output those within the list
    '''
    tag_id = 0
    tag2id = {}
    out_bams = {}
    futils.good_prefix(bam_path, futils.out_prefix)
    process_tag_file(todo='init', out_dir=out_dir)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for aln in bam:
            aln_tag = get_tags(aln, tag_id_list=[tag_demux])
            if not aln_tag:
                continue
            if aln_tag not in tag2id:
                this_id = tag_id
                tag2id[aln_tag] = this_id
                tag_file = os.path.join(out_dir, f'{futils.out_prefix}_{aln_tag}.bam')
                out_bams[this_id] = pysam.AlignmentFile(tag_file, 'wb', template=bam)
                out_bams[this_id].write(aln)
                tag_id += 1
            else:
                this_id = tag2id[aln_tag]
                out_bams[this_id].write(aln)
            process_tag_file(todo='write', aln=aln)

    logging.info(f'{tag_id} tags in total')
    [out_bams[b].close() for b in out_bams]
    process_tag_file(todo='close')
    [pysam.index(out_bams[b]) for b in out_bams]

def subset_bam(bam_path, out_dir, read_id_path):
    '''
    extract alignments by read ids
    Inputs:
        - read_id_path: text file with read ids (no header)
    '''
    reads = futils.file2set(read_id_path, sep='\t', val_col=0)
    reads = set(reads)
    num_reads = len(reads)
    logging.info(f'{num_reads} reads in total...')

    logging.info('Filtering bam...')
    #futils.good_prefix(bam_path, futils.out_prefix, add2last=".sub")
    out = os.path.join(out_dir, f'sub_{os.path.basename(bam_path)}')

    read_cnt = 0
    with pysam.AlignmentFile(bam_path, 'rb') as bam_in:
        with pysam.AlignmentFile(out, 'wb', template=bam_in) as bam_out:
            for aln in bam_in:
                if aln.query_name in reads:
                    bam_out.write(aln)
                    read_cnt += 1
                if read_cnt == num_reads:
                    break
    pysam.sort("-o", out.replace('.bam', '.sorted.bam'), out, "-@", "4")
    pysam.index(out.replace('.bam', '.sorted.bam'), "-@", "4")
    os.remove(out)

def get_filter_tag_list(filter_tag, filter_tag_file):
    filter_tag_list = futils.file2set(
        filter_tag_file, 
        sep='\t', 
        val_col=0, 
        header_lines=None)
    assert filter_tag is not None, 'What tag? Please specify one tag id.'
    assert len(filter_tag.split(',')) == 1, 'only support filter for one tag now'
    return set(filter_tag_list)

def get_tags(aln, tag_id_list):
    tags = [i[1] for i in aln.tags if i[0] in tag_id_list]
    # output string
    if len(tags) == 0:
        return None
    else:
        return ','.join(tags)

def rev_complement(seq):
    matches = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N'}
    rev_seq = ''.join([matches[nt] for nt in reversed(seq)])
    return rev_seq

def edit_distance(s1, s2):
    return ed.eval(s1, s2)

def hamming_distance(s1, s2): 
    assert len(s1) == len(s2), 'string lengths are not the same'
    distance = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            distance += 1
    return distance

def compute_seq_entropy(seq):
    base_frq = {base: seq.count(base) / len(seq) for base in set(seq)}
    entropy = -sum(frq * np.log2(frq) for frq in base_frq.values() if frq > 0)
    return entropy

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def search_kmers(seq, target_seq, k, max_ed):
    # calculate ed dist in each iteration, stop at the first match
    target_pos = None
    for i in range(len(seq)-k+1):
        this_kmer = seq[i:i+k]
        if edit_distance(this_kmer, target_seq) <= max_ed:
            target_pos = i
            break
    return target_pos

def save_tags(aln, tag_list, out):
    tags = [f'{i[0]}:{i[1]}' for i in aln.tags if i[0] in tag_list]
    row = futils.list2line([aln.query_name] + tags)
    out.write(row.encode())

def read_tag_file(tag_file, tag_names):
    '''
    format: id tag1 tag2 tag3 tag4 ...
    '''
    read2tags = defaultdict(list)
    tag_names = tag_names.split(',')

    with futils.open_text(tag_file) as f:
        for line in f:
            l = futils.read_line(line)
            read2tags[l[0]] = dict((tag_names[i], t) for i,t in enumerate(l[1:]))
    return read2tags

def process_tag_file(todo, tag_list, save_tags_to, out_dir=None, aln=None):
    if todo == 'init':
        if not out_dir:
            sys.exit('out_dir is None!')
        save_tags_to = futils.write_text(
            os.path.join(out_dir, f'{futils.out_prefix}.tags.tsv.gz')
        )
    elif todo == 'write':
        save_tags(aln, tag_list, save_tags_to)
    elif todo == 'close':
        save_tags_to.close()

READ_HEADERS = {
    'feature_table': [
        'barcode', 'umi', 'read_id', 'feature', 'feature_id', 'feature_name'
    ]
}

@dataclass
class Read:
    id: str
    chr: str
    start: int # 1-based, pysam is 0-based
    end: int
    len: int
    mapq: int
    cigar: str
    cb: str
    umi: str
    strand: str = '.'
    struct: str = None # Categories: M (monoexonic), S (fully spliced), U (unspliced in >=1 introns)
    feature: str = '.' # gene, alu, alu_oth, intergenic, intron
    mismatch: dict = field(default_factory=dict) # {pos:allele}
    gene: dict = field(default_factory=dict) # {id, name, tx}
    repeat: dict = field(default_factory=dict) # {pos, fam, overlap}
    hap: dict = field(default_factory=dict) # {loc:hap_k}

    @property
    def is_monoexonic(self):
        return self.cigar_info()

    def cigar_info(self, max_err=5):
        '''
        remove two ends in case weird things happend
        if events < 2, then it's probably monoexnoic (gapless)
        '''
        cigartuples = re.findall(r'(\d+)([A-Za-z])?', self.cigar)
        events = [t[1] for t in cigartuples[1:-1] if int(t[0]) >= max_err]
        mono = 'N' not in events
        #num_clippings = sum([int(t[0]) for t in cigartuples if t[1] in ['S', 'H']])
        return mono

    @property
    def is_from_genic_region(self):
        # genic means if this read is located within a gene
        return self.from_gene_region()

    def from_gene_region(self):
        if 'id' not in self.gene and 'name' not in self.gene:
            return False
        return True

    def get_feature_to_count(self):
        '''
        Further classify repeat reads into genic & intergenic
        '''
        if self.feature in ['gene', 'intron', 'intergenic', 'an']:
            return self.feature
        
        if self.is_from_genic_region:
            return f'{self.feature}_genic'
        else:
            return f'{self.feature}_intergenic'

    def get_strand(self, genic=True, stranded=False):
        assert self.feature is not None, 'Annotate reads first'
        if stranded:
            return self.strand
        
        if genic and self.feature != 'gene':
            return '.'
        else:
            return self.strand

    @classmethod
    def same_aln(cls, read, aln):
        if aln.reference_name == read.chr and \
            aln.reference_start+1 == read.start and \
            aln.cigarstring == read.cigar:
                return True
        return False

    def extract_exon_blocks(self, cigartuples):
        # all positions are 1-base
        # NOTE cigartuples from pysam segment
        max_del_len = 10
        start = end = self.start
        blocks = [] # [(st, ed)]
        for event in cigartuples:
            if event[0] in [1, 4, 5]:
                # insertion, soft & hard clippings
                continue
            elif event[0] == 0:
                # mapped
                end += event[1]
            elif event[0] == 2:
                # deletion from reference
                if event[1] > max_del_len:
                    blocks.append((start, end))
                    start = end + event[1]
                    end = start
                else:
                    end += event[1]
            elif event[0] == 3:
                # intron
                blocks.append((start, end))
                start = end + event[1]
                end = start
            else:
                raise ValueError('Found unparsed cigar string: ' + str(event[0]))
        blocks.append((start, end))
        return blocks

    def write_bed(self, cigartuples, other_info):
        '''generate rows in bed format
        - other_info: list of other info to put in bed
        output: chrom start end exon_id exon_len
        '''
        all_rows = []
        if self.is_monoexonic:
            # NOTE there could be disconcordance between end-start and length bc of indels
            exon_id = 'exon1'
            bed_row = [self.chr, self.start, self.end, exon_id, self.len] + other_info
            all_rows.append(bed_row)
        else:
            # for spliced reads just take exons
            ## NOT consider strand ##
            exon_blocks = self.extract_exon_blocks(cigartuples)
            for idx, exon in enumerate(exon_blocks):
                exon_start, exon_end = exon
                if idx == len(exon_blocks) - 1:
                    exon_id = 'last_exon'
                else:
                    exon_id = f'exon{idx+1}'
                exon_len = exon_end - exon_start
                bed_row = [self.chr, exon_start, exon_end, exon_id, exon_len] + other_info
                all_rows.append(bed_row)
        return all_rows
    
    def get_feature_id_name(self, feature='gene'):
        feature_id = feature_name = '.'
        
        if feature == 'gene':
            if 'id' in self.gene:
                feature_id = self.gene['id']
                feature_name = self.gene['name']
                if ';' in self.gene['id']:
                    picked_idx = random.choice(range(len(self.gene['id'].split(';'))))
                    feature_id = self.gene['id'].split(';')[picked_idx]
                    feature_name = self.gene['name'].split(';')[picked_idx]
                feature_name = feature_name.split('|')[0]
            elif 'pos' in self.repeat:
                feature_id = feature_name = self.repeat['pos']
        elif feature == 'transcript' and 'tx' in self.gene:
            feature_id = feature_name = self.gene['tx']
            if ';' in self.gene['tx']:
                picked_idx = random.choice(range(len(self.gene['tx'].split(';'))))
                feature_id = feature_name = self.gene['tx'].split(';')[picked_idx]
                feature_name = feature_id
        return feature_id, feature_name

    def write_read_info(self):
        '''
        generate read to feature info, for outputting read info file and getting cell matrix
        '''
        feature_id, feature_name = self.get_feature_id_name()
        items = [self.cb, self.umi, self.id, self.len, self.strand, self.feature, feature_id, feature_name]
        return futils.list2line(items)
    
    @classmethod
    def get_ref_pos(cls, ref_start, cigartuples):
        '''
        CIGAR:
         "M": 0
         "I": 1
         "D": 2
         "N": 3
         "S": 4
         "H": 5
         "P": 6
         "=": 7
         "X": 8
        
        Map query base to reference coordinates (1-based). 
        Returns 
            list with len == seq_len
        '''
        ref_pos = []
        s = ref_start
        for event, l in cigartuples:
            if event in [2,3]:
                s += l
            elif event in [1,4]:
                pos_list = [None] * l
                ref_pos.extend(pos_list)
            elif event in [5,6]:
                continue
            elif event in [0,7,8]:
                # mapped
                pos_list = list(range(s, s + l))
                ref_pos.extend(pos_list)
                s += l
            else:
                raise ValueError(f'new cigar event {event}, not parsed')
        return ref_pos
