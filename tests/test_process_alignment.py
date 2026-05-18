from mola.read.process_alignment import Read


def make_read():
    return Read(
        id='read1',
        chr='chr1',
        start=101,
        end=151,
        len=50,
        mapq=60,
        cigar='50=',
        cb='.',
        umi='.',
    )


def test_extract_exon_blocks_accepts_pacbio_match_op():
    read = make_read()

    assert read.extract_exon_blocks([(7, 50)]) == [(101, 151)]


def test_extract_exon_blocks_keeps_pacbio_match_and_mismatch_contiguous():
    read = make_read()

    assert read.extract_exon_blocks([(7, 20), (8, 5), (7, 25)]) == [(101, 151)]


def test_extract_exon_blocks_splits_pacbio_alignment_on_introns():
    read = make_read()

    assert read.extract_exon_blocks([(7, 20), (3, 100), (8, 25)]) == [
        (101, 121),
        (221, 246),
    ]


def test_extract_exon_blocks_keeps_short_deletions_inside_exon():
    read = make_read()

    assert read.extract_exon_blocks([(7, 20), (2, 5), (8, 25)]) == [(101, 151)]


def test_extract_exon_blocks_splits_long_deletions():
    read = make_read()

    assert read.extract_exon_blocks([(7, 20), (2, 20), (8, 25)]) == [
        (101, 121),
        (141, 166),
    ]
