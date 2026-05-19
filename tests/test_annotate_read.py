from mola.read.annotate_read import dup_mole, make_unique_read_id
from mola.read.process_alignment import Read


def make_read(read_id='read1', start=101, length=50, cb='cell1', umi='umi1'):
    return Read(
        id=read_id,
        chr='chr1',
        start=start,
        end=start + length,
        len=length,
        mapq=60,
        cigar=f'{length}M',
        cb=cb,
        umi=umi,
    )


def test_make_unique_read_id_keeps_first_name_and_suffixes_repeats():
    read_id_counts = {}

    assert make_unique_read_id('molecule/20086815', read_id_counts) == 'molecule/20086815'
    assert make_unique_read_id('molecule/20086815', read_id_counts) == 'molecule/20086815/2'
    assert make_unique_read_id('molecule/20086815', read_id_counts) == 'molecule/20086815/3'


def test_dup_mole_skips_reads_without_barcode_or_umi():
    uniq_molecules = {}
    reads_chrom = {}

    action = dup_mole(
        make_read(read_id='molecule/20086815', cb='.', umi='.'),
        uniq_molecules,
        reads_chrom,
        max_dist=1,
    )

    assert action == 'new'
    assert uniq_molecules == {}


def test_dup_mole_recovers_from_stale_duplicate_pointer():
    uniq_molecules = {('cell1', 'umi1'): {101: 'molecule/20086815'}}
    reads_chrom = {}

    action = dup_mole(
        make_read(read_id='molecule/20086816', start=101),
        uniq_molecules,
        reads_chrom,
        max_dist=1,
    )

    assert action == 'new'
    assert uniq_molecules == {('cell1', 'umi1'): {101: 'molecule/20086816'}}


def test_dup_mole_keeps_same_barcode_umi_reads_at_distant_starts():
    old_read = make_read(read_id='read-old', start=101)
    uniq_molecules = {('cell1', 'umi1'): {old_read.start: old_read.id}}
    reads_chrom = {old_read.id: old_read}

    action = dup_mole(
        make_read(read_id='read-new', start=501),
        uniq_molecules,
        reads_chrom,
        max_dist=1,
    )

    assert action == 'new'
    assert uniq_molecules == {('cell1', 'umi1'): {101: 'read-old', 501: 'read-new'}}


def test_dup_mole_replaces_nearby_shorter_read():
    old_read = make_read(read_id='read-old', start=101, length=50)
    uniq_molecules = {('cell1', 'umi1'): {old_read.start: old_read.id}}
    reads_chrom = {old_read.id: old_read}

    action = dup_mole(
        make_read(read_id='read-new', start=102, length=60),
        uniq_molecules,
        reads_chrom,
        max_dist=1,
    )

    assert action == 'replace'
    assert reads_chrom == {}
    assert uniq_molecules == {('cell1', 'umi1'): {102: 'read-new'}}
