"""Generate genome browser files for plotting with coolbox"""

import pyBigWig
import coolbox
from coolbox.api import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from nmf_models_mod_updates import intNMF
from typing import Optional, Union, Mapping  # Special
import seaborn as sns
import matplotlib as mpl
import sys
import pandas as pd

def get_chrom_sizes(genome_file_loc):
    with open(genome_file_loc) as f:
        chrom_sizes = []
        for i in f.readlines():
            tmp = i.split("\t")
            chrom_sizes.append((tmp[0],int(tmp[1])))
    return chrom_sizes

def select_valid_chroms(data, chrom_sizes, ignore=['_', 'M']):

    valid_chroms = [i[0] for i in chrom_sizes if not any(sub_str in i[0] for sub_str in ['_', 'M'])]
    idxs_keep = []
    for i, chro in enumerate(data[0]):
        if chro in valid_chroms:
            idxs_keep.append(i)
    
    return [d[idxs_keep] for d in data], valid_chroms

def get_feature_locations(nmf_model, k , mode):
    if mode == 'atac':
        chroms_np = np.array([region.split(':')[0] for region in nmf_model.atac_features])
        starts_np = np.array([region.split(':')[1].split('-')[0] for region in nmf_model.atac_features], dtype='int')
        ends_np = np.array([region.split(':')[1].split('-')[1] for region in nmf_model.atac_features], dtype='int')
        scores_np = nmf_model.phi_atac[k, :]
    elif mode == 'rna':
        chroms_np = np.array([gene.split(':')[0] for gene in nmf_model.rna_features])
        starts_np = np.array([gene.split(':')[1].split('-')[0] for gene in nmf_model.rna_features], dtype='int')
        ends_np = np.array([gene.split(':')[1].split('-')[1] for gene in nmf_model.rna_features], dtype='int')
        scores_np = nmf_model.phi_rna[k, :]
    else:
        print('select mode from "rna" or "atac"')
        assert False

    return [chroms_np, starts_np, ends_np, scores_np]

def make_bigwig(file_name, nmf_model, k, genome_file_loc, mode = 'atac'):

    
    data = get_feature_locations(nmf_model, k, mode)

    chrom_sizes = get_chrom_sizes(genome_file_loc)

    data, valid_chroms = select_valid_chroms(data, chrom_sizes)

    bw = pyBigWig.open(file_name, 'w')
    bw.addHeader(chrom_sizes)

    for chr in valid_chroms:
        idx = np.nonzero(data[0] == chr)
        data_chr = [d[idx] for d in data]

        order = np.argsort(data_chr[1])
        data_ord = [d[order] for d in data_chr]

        chroms_s = [str(x) for x in data_ord[0]]
        starts_i = [int(x) for x in data_ord[1]]
        ends_i = [int(x) for x in data_ord[2]]
        scores_f = [float(x) for x in data_ord[3]]
        
        current = [chroms_s[0], starts_i[0], ends_i[0], scores_f[0]]

        for c, s, e, sc in zip(chroms_s, starts_i, ends_i, scores_f):
            last = current
            current = [c, s, e, sc]
            #try:
            bw.addEntries([c],[s], ends=[e], values=[sc])
            #except:
            #    print("Error\nLast added entry: {}".format(last))
            #    print("Failed to add entry: {}".format(current))
        
    bw.close()