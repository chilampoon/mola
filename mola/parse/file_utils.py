"helper functions for file io/parsing"

import gzip
import os
import re
import logging
import sys
import subprocess
import pickle
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import tempfile
import shutil
import itertools
import glob
import random
import platform
from collections import defaultdict
import concurrent.futures

class FileIO:
    def __init__(self):
        self.is_gz = None
        self.header = None
        self.sep = None
        self.force_rewrite = True
        self.startswith_chr = True

    def is_gz_file(self, file):
        with open(file, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'

    def open_text(self, file, mode='r', sep='\t'):
        self.sep = sep
        if self.is_gz_file(file):
            self.is_gz = True
            if 'b' not in mode:
                mode += 'b'
            return gzip.open(file, mode=mode)
        else:
            self.is_gz = False
            return open(file, mode=mode)

    def header_dict(self, file):
        # file is opened
        for _ in range(1):
            l = file.readline()
            header = self.read_line(l, sep=self.sep)
        header_dict = dict((x, i) for i, x in enumerate(header))
        self.header = header_dict
        return header_dict

    def read_line(self, line, sep='\t'):
        if sep != self.sep:
            self.sep = sep
        if self.is_gz:
            r = line.decode().rstrip().split(sep)
        else:
            r = line.rstrip().split(sep)
        return r
    
    def list2line(self, lst, sep='\t'):
        return sep.join(map(str, lst)) + '\n'

    def write_text(self, file, mode='w'):
        # remember to close the file later on
        ## first check if file exists, if so, exit
        if os.path.isfile(file):
            if not self.force_rewrite:
                logging.warning(f'{file} exists, exiting. Can change rewrite setting.')
                sys.exit(1)
            else:
                logging.warning(f'{file} exists, rewriting!')
        
        if file.endswith('.gz'):
            if 'b' not in mode:
                mode += 'b'
            return gzip.open(file, mode=mode)
        else:
            return open(file, mode=mode)

    def good_prefix(self, file=None, out_prefix=None, add2last=''):
        '''prefix here is not ending with any symbols'''
        if out_prefix:
            new_prefix = f'{out_prefix}{add2last}'
        elif file:
            name, extension = os.path.splitext(file)
            new_prefix = os.path.basename(file).replace(extension, add2last)
        else:
            new_prefix = 'longr_out'      
        self.out_prefix = new_prefix
    
    def make_tmp_dir(self, out_dir, suffix=None):
        tmp_dir = tempfile.mkdtemp(dir=out_dir, suffix=suffix if suffix else "")
        return tmp_dir
    
    def make_tmp_file(self, out_dir, suffix='txt', is_gz=True):
        # just need a name
        if is_gz:
            mode = "w+b"
            suffix = f'{suffix}.gz'
        else:
            mode = "w"
            suffix = suffix
        out = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=True, dir=out_dir)
        return out.name
    
    def load_gz_pickle(self, file):
        with gzip.open(file, 'rb') as f:
            return pickle.load(f)
    
    def save_gz_pickle(self, file, obj):
        with gzip.open(file, 'wb') as f:
            pickle.dump(obj, f)


class FileUtils(FileIO):
    def __init__(self):
        super().__init__()

    def check_chr_id(self, chrom):
        if self.startswith_chr and not chrom.startswith('chr'):
            new_chr = 'chr' + chrom
        elif not self.startswith_chr and chrom.startswith('chr'):
            new_chr = re.sub('^chr', '', chrom)
        else:
            new_chr = chrom
        return new_chr
    
    def sort_chroms(self, chroms):
        return sorted(chroms, key=self.sort_any_chroms)
        
    def sort_any_chroms(self, s):
        '''sorts chromosome names including traditional chr names and more general identifiers.
        '''
        num_part = re.search(r'\d+', s)
        
        if num_part:
            return int(num_part.group())
        else:
            if s in ['chrX', 'chrY', 'chrM', 'chrMT']:
                return float('inf') + ord(s[-1])
            else:
                # for other non-numeric strings, sort them alphabetically
                return float('inf') + sum(ord(char) for char in s)
        
    def sort_chrom_files(self, path):
        # file: chr1.xxx.gz, usually in tmp dir
        chrom = path.split('/')[-1].split('.')[0]
        return self.sort_any_chroms(chrom)
    
    def file2set(self, file, sep, val_col=0, header_lines=None):
        '''
        read file, extract info from 1 column,
        and save into a set
        (val_col)
        '''
        if header_lines is not None:
            assert type(header_lines) == int, 'input header_lines an interger'
        out_set = set()
        with self.open_text(file, sep=sep) as f:
            if header_lines is not None:
                for _ in range(header_lines):
                    next(f)
            for line in f:
                l = self.read_line(line, sep=sep)
                out_set.add(l[val_col])
        return out_set

    def file2dict(self, file, key_col, val_col, sep='\t',
                  val_type=set, header_lines=None):
        '''
        Read file, extract info and save into a dict
        Now it's simple:
        {key_col: val_col}
        Inputs:
          - key_col: the number of column containing keys
          - val_col: the number of column cotaining values
          - val_type: type of object for stored values
        '''
        out_dict = defaultdict(val_type)
        with self.open_text(file) as f:
            if header_lines is not None:
                for _ in range(header_lines):
                    next(f)
            for line in f:
                l = self.read_line(line, sep)
                out_dict[l[key_col]].add(l[val_col])
        return out_dict
    
    def is_command_avail(self, command):
        '''test if a software is installed'''
        try:
            result = subprocess.run(
                ['which', command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False

    def process_chrom_in_threads(self, func, chrom_list, num_threads=None, **kwargs):
        '''
        processes given function in parallel or sequentially based on num_threads.
        - func: function to process.
        - chrom_list: a list of chromosome ids
        - num_threads: number of threads for parallel processing. If None, will process sequentially.
        - kwargs: additional keyword arguments to pass to the function.
        '''
        results = {}
        
        if num_threads is not None:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # submit tasks to thread pool
                    futures = {executor.submit(func, chrom, **kwargs): chrom for chrom in chrom_list}
                    concurrent.futures.wait(futures)
                    
                    for future in concurrent.futures.as_completed(futures):
                        chrom = futures[future]
                        if future.exception() is not None:
                            logging.error(f"Exception from {chrom}: {future.exception()}")
                        else:
                            results[chrom] = future.result()
            except Exception as e:
                logging.error(f"Error occurred in process_in_threads: {e}")
        else:
            # no parallel, run one by one
            for chrom in chrom_list:
                results[chrom] = func(chrom, **kwargs)

        return results
    
    def concat_files(self, sorted_file_list, out):
        '''all files shouldn't have a header,
           the header has been printed to the final output file 
        '''
        cmd = ['cat'] + sorted_file_list + ['>>', out]
        subprocess.run(' '.join(cmd), shell=True)

    def write_base_script(self, script, out):
        with self.write_text(out) as f:
            f.write(script)
        os.chmod(out, 0o755)
