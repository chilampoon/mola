from mola.read.process_alignment import *
import random

BASEORDER = ['A', 'C', 'G', 'T', 'N']
REV_COMP_BASES = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'N':'N', '.':'.'}
MIN_GENIC_READ_RATIO = 0.85
MAX_GAPLESS_READ_RATIO = 0.8
GENIC_REG_ORDER = [
    'CDS', 'start_codon', 'Selenocysteine', 'stop_codon', 'last_exon', 
    'first_exon', 'exon', 'five_prime_UTR', 'three_prime_UTR', 'UTR', 'intron'
]


def output_site_table(site_dir, stranded, mode, out_path, celltype_map):
    modes = ['bulk', 'pseudobulk', 'cell']
    if mode not in modes:
        raise ValueError("Invalid mode. Expected one of: %s" % modes)
    if mode != 'bulk' and not celltype_map:
        raise ValueError(f"cell_to_celltype mapping required for '{mode}' mode")
    chroms = fetch_chrom_ids(site_dir)

    # set up header
    cell2celltype = cell_to_celltype(celltype_map) if celltype_map else None
    header = SITE_HEADERS['stranded_site_table'] if stranded else SITE_HEADERS['unstranded_site_table']
    if mode == 'pseudobulk':
        header = ['celltype'] + header
    elif mode == 'cell':
        header = ['celltype', 'barcode'] + header if celltype_map else ['barcode'] + header
    header = '\t'.join(header) + '\n'

    with futils.write_text(out_path) as out:
        out.write(header.encode())
        for chrom in chroms:
            logging.info(f'Writing {chrom}...')
            obj_path = os.path.join(site_dir, f'{chrom}_sites.pkl.gz')
            with gzip.open(obj_path) as obj:
                sites = pickle.load(obj)
                for k,v in sites.items():
                    row = v.write_table(stranded=stranded, cell_to_celltype=cell2celltype, mode=mode)
                    out.write(row.encode())

def output_cell_by_site_matrix(site_dir, out_dir, celltype_map, ref='A', alt='G'):
    '''Output cell by site matrix
    NOTE: rn specified ref allele is A alt is G!!
    '''
    chroms = fetch_chrom_ids(site_dir)
    cell2celltype = cell_to_celltype(celltype_map) if celltype_map else None
    out_path_ref = os.path.join(out_dir, f'cell_by_site_ref.tsv.gz')
    out_path_alt = os.path.join(out_dir, f'cell_by_site_alt.tsv.gz')
    
    with futils.write_text(out_path_ref) as fref, futils.write_text(out_path_alt) as falt:
        fref.write(futils.list2line(['#SNV'] + [cell for cell in cell2celltype]).encode())
        falt.write(futils.list2line(['#SNV'] + [cell for cell in cell2celltype]).encode())
        for chrom in chroms:
            snv2cell = {}
            logging.info(f'Counting mismatches in {chrom}...')
            sites = futils.load_gz_pickle(f'{site_dir}/{chrom}_sites.pkl.gz')
            for pos, site in sites.items():
                snv_id = f'{site.chrom}:{site.pos}'
                snv2cell[snv_id] = defaultdict(lambda: {'ref':0, 'alt':0})
                if site.strand == '-':
                    ref, alt = REV_COMP_BASES[ref], REV_COMP_BASES[alt]
                
                for cb, read_info in zip(site.cells, site.reads):
                    cb = re.sub(r'-[0-9]+', '', cb)
                    read_id, read_strand, allele = read_info.split('|')
                    if allele == ref:
                        snv2cell[snv_id][cb]['ref'] += 1
                    elif allele == alt:
                        snv2cell[snv_id][cb]['alt'] += 1
            # write to file
            for snv_id, cell2cnts in snv2cell.items():
                fref.write(futils.list2line([snv_id] + [cell2cnts[cell].get('ref', 0) for cell in cell2celltype]).encode())
                falt.write(futils.list2line([snv_id] + [cell2cnts[cell].get('alt', 0) for cell in cell2celltype]).encode())

def fetch_chrom_ids(site_dir):
    # fetch all chrom names
    chrom_files = glob.glob(os.path.join(site_dir, '*_sites.pkl.gz'))
    chroms = [os.path.basename(f).split('_sites.pkl.gz')[0] for f in chrom_files]
    chroms = futils.sort_chroms(chroms)
    return chroms

def cell_to_celltype(celltype_map):
    '''
    celltype_map: text file (barcode \t celltype)
    '''
    cell2celltype = defaultdict(str)
    with futils.open_text(celltype_map) as f:
        #next(f)
        for line in f:
            line = futils.read_line(line)
            cell = re.sub(r'-[0-9]+', '', line[0])
            cell2celltype[cell] = line[1]
    return cell2celltype


@dataclass
class MismatchSite:
    chrom: str
    pos: int
    mismatch: str # all forward
    total: int
    base_cnts: dict = field(default_factory=lambda: {b:0 for b in BASEORDER})
    major_cnt: int = None
    minor_cnt: int = None # max of minor cnts
    read_comp: dict = None
    read_spliced: dict = field(default_factory=lambda: {i:0 for i in ['U', 'S']})
    strand: str = None # from the major one
    snp: str = None
    repeat: set = field(default_factory=set)
    gene: dict = field(default_factory=dict)
    genic: bool = None
    reads: list = field(default_factory=list) # format: read_id|strand|base
    cells: list = field(default_factory=list) # same length as reads
    posteriors: dict = None # unstranded

    @property
    def major_minor_stats(self):
        # pseudobulk/bulk stats
        if not all([self.major_cnt, self.minor_cnt]):
            self.major_cnt, self.minor_cnt, self.major_frq, self.minor_frq = self.calc_major_minor_stats()
        return self.major_cnt, self.minor_cnt, self.major_frq, self.minor_frq

    def mismatch_from_cnt(self):
        major_ales = [b for b in BASEORDER if self.base_cnts[b] == self.major_cnt]
        if len(major_ales) > 1:
            # randomly pick one allele for major
            major_ale = random.choice(major_ales)
            minor_ale = random.choice([a for a in major_ales if a != major_ale])
        else:
            major_ale = major_ales[0]
            minor_ale = [b for b in BASEORDER if self.base_cnts[b] == self.minor_cnt][0]
        return f'{major_ale}>{minor_ale}'
    
    def calc_major_minor_stats(self, cnt_lst=None):
        if not cnt_lst:
            cnt_lst = list(self.base_cnts.values())
        cnt_lst.sort()
        cnt_lst = cnt_lst[::-1]
        major_cnt, minor_cnt = cnt_lst[0], cnt_lst[1]
        major_frq = major_cnt / self.total
        minor_frq = minor_cnt / self.total
        return major_cnt, minor_cnt, major_frq, minor_frq

    @property
    def get_strand(self):
        if self.strand is None:
            self.strand = self.major_strand()
        return self.strand

    def major_strand(self):
        strands = [r.split('|')[1] for r in self.reads]
        return max(strands, key=lambda k: strands.count(k))

    @property
    def mismatch_type_by_strand(self):
        if self.strand == '-':
            mismatch_by_strand = self.rev_compl_mismatch(self.mismatch)
        else:
            mismatch_by_strand = self.mismatch
        return mismatch_by_strand
    
    def mismatch_id(self):
        return f'{self.chrom}:{self.pos}|{self.mismatch}'

    def on_alu(self):
        if sum(['Alu' in region for region in self.repeat]) > 0:
            return True
        return False

    @classmethod
    def rev_compl_mismatch(cls, mismatch):
        return '>'.join([REV_COMP_BASES[b] for b in mismatch.split('>')])
    
    @property
    def is_from_genic(self):
        if self.genic is None:
            self.genic = self.from_genic_reads()
        return self.genic

    def homopolymer_check(self, ref_fa, window=3):
        '''
        Check if the mismatch is within homopolymer region
        using reference sequence within pos +/- window
        '''
        with pysam.FastaFile(ref_fa) as fa:
            lhs_seq = fa.fetch(self.chrom, self.pos-1-window, self.pos-1)
            rhs_seq = fa.fetch(self.chrom, self.pos, self.pos+window)
            lhs_seq = lhs_seq.upper(); rhs_seq = rhs_seq.upper()
            # how poly is it
            if len(set(lhs_seq)) == len(set(rhs_seq)) == 1 and set(lhs_seq) == set(rhs_seq):
                return True
        return False

    def from_genic_reads(self):
        '''
        Determine if the mismatch is from genic reads (reads with isoform structure);
        if it is from a gene, then the strand is known.
        For long reads only.
        '''
        if not self.gene:
            return False
        if 'gene' not in self.read_comp:
            return False

        genic_read_prop = self.read_comp['gene'] / sum(self.read_comp.values())
        gapless_read_prop = self.read_spliced['U'] / sum(self.read_spliced.values())

        is_genic = genic_read_prop >= MIN_GENIC_READ_RATIO
        excessively_gapless = gapless_read_prop >= MAX_GAPLESS_READ_RATIO
        is_on_alu = self.on_alu()
        return is_genic and not (excessively_gapless and is_on_alu)

    def get_genic_region(self):
        if 'reg' in self.gene:
            reg = set([r.split('@')[0] for r in self.gene['reg']])
            sorted_reg = sorted(reg, key=lambda x: GENIC_REG_ORDER.index(x) if x in GENIC_REG_ORDER else len(GENIC_REG_ORDER))
            final_reg = sorted_reg[0]
            if final_reg == 'first_exon':
                final_reg = '5UTR'
            elif final_reg == 'last_exon':
                final_reg = '3UTR'
            return final_reg
        else:
            return '.'
    
    def get_gene_name(self):
        if 'name' in self.gene:
            gene_name = self.gene['name']
            if ';' in gene_name and self.strand != '.':
                gene_name_split = [n for n in gene_name.split(';') if n.split('|')[1]==self.strand]
                if len(gene_name_split) > 0:
                    gene_name = gene_name_split[0]
            return gene_name
        else:
            return '.'

    def write_bed(self):
        return f'{self.chrom}\t{self.pos-1}\t{self.pos}\n'

    def write_table(self, stranded = False, cell_to_celltype = None, mode='bulk'):
        # stranded means stranded output, not stranded reads
        # cell_to_celltype is a dict mapping cell barcode to celltype
        
        strand = self.get_strand
        gene_name = self.get_gene_name()
        gene_region = self.get_genic_region()
        if stranded:
            mismatch_by_strand = self.mismatch_type_by_strand

        if mode == 'bulk':
            # bulk on whole transcriptome
            major_cnt, minor_cnt, _, minor_frq = self.major_minor_stats
            counts = list(map(str, [self.total, major_cnt, minor_cnt, minor_frq]))
            if stranded:
                row = [self.mismatch_id(), strand, mismatch_by_strand] + list(map(str, self.base_cnts.values())) + \
                    counts + [self.snp, gene_name, gene_region]
            else:
                row = [self.mismatch_id(), self.mismatch] + list(map(str, self.base_cnts.values())) + \
                        counts + [self.snp, gene_name, gene_region]
            row = '\t'.join(row) + '\n'
        elif mode == 'pseudobulk':
            # pseudobulk on cell type
            celltype_to_cnts = defaultdict(lambda: {b:0 for b in BASEORDER})
            for cell, read in zip(self.cells, self.reads):
                cell = re.sub(r'-[0-9]+', '', cell) # just in case
                if cell not in cell_to_celltype:
                    continue
                celltype = cell_to_celltype[cell]
                base = read.split('|')[-1]
                celltype_to_cnts[celltype][base] += 1
            
            major_allele, minor_allele = self.mismatch.split('>')
            row = ''
            for celltype in celltype_to_cnts:
                # here the major_cnt/minor_cnt should be count for major/minor allele
                count_lst = list(celltype_to_cnts[celltype].values())
                total = sum(count_lst)
                major_cnt = celltype_to_cnts[celltype][major_allele]
                minor_cnt = celltype_to_cnts[celltype][minor_allele]
                minor_frq = minor_cnt / total
                counts = list(map(str, [total, major_cnt, minor_cnt, minor_frq]))
                if stranded:
                    new_row = [celltype, self.mismatch_id(), strand, mismatch_by_strand] + list(map(str, count_lst)) + \
                        counts + [self.snp, gene_name, gene_region]
                else:
                    new_row = [celltype, self.mismatch_id(), self.mismatch] + list(map(str, count_lst)) + \
                            counts + [self.snp, gene_name, gene_region]
                row += ('\t'.join(new_row) + '\n')
        elif mode == 'cell':
            # counts at the cell level
            cell_to_cnts = defaultdict(lambda: {b:0 for b in BASEORDER})
            for cell, read in zip(self.cells, self.reads):
                cell = re.sub(r'-[0-9]+', '', cell) # just in case
                if cell not in cell_to_celltype:
                    continue
                base = read.split('|')[-1]
                cell_to_cnts[cell][base] += 1
                
            major_allele, minor_allele = self.mismatch.split('>')
            row = ''
            for cell in cell_to_cnts:
                count_lst = list(cell_to_cnts[cell].values())
                total = sum(count_lst)
                major_cnt = cell_to_cnts[cell][major_allele]
                minor_cnt = cell_to_cnts[cell][minor_allele]
                minor_frq = minor_cnt / total
                counts = list(map(str, [total, major_cnt, minor_cnt, minor_frq]))
                if stranded:
                    new_row = [cell, self.mismatch_id(), strand, mismatch_by_strand] + list(map(str, count_lst)) + \
                        counts + [self.snp, gene_name, gene_region]
                else:
                    new_row = [cell, self.mismatch_id(), self.mismatch] + list(map(str, count_lst)) + \
                            counts + [self.snp, gene_name, gene_region]
                if cell_to_celltype:
                    celltype = cell_to_celltype[cell] if cell in cell_to_celltype else '.'
                    new_row = [celltype] + new_row
                row += ('\t'.join(new_row) + '\n')
        return row

SITE_HEADERS = {
    'stranded_site_table': [
        'snv', 'strand', 'mismatch'] + BASEORDER + ['total', 'major_cnt', 'minor_cnt', 'minor_af', 'snp_type', 'gene', 'region'
    ],
    'unstranded_site_table': [
        'snv', 'mismatch'] + BASEORDER + ['total', 'major_cnt', 'minor_cnt', 'minor_af', 'snp_type', 'gene', 'region'
    ],
}
