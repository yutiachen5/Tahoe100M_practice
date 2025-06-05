from datasets import load_dataset
import json
import gzip

tahoe_100m_ds = load_dataset('vevotx/Tahoe-100M', streaming=True, split='train')
N = 10_000_000  
i = 0

with gzip.open('/hpc/group/naderilab/eleanor/prose_data/data/Tahoe_downsampled.txt.gz', 'wt', encoding='utf-8') as outfile:
    outfile.write('genes\texpressions\tcanonical_smiles\tdrug\tcell_line_id\n')
    for record in tahoe_100m_ds:
        if i >= N: 
            break
        genes_str = ','.join(map(str, record['genes']))
        expressions_str = ','.join(map(str, record['expressions']))
        line = '\t'.join([genes_str, expressions_str, record['canonical_smiles'], record['drug'], record['cell_line_id']])
        outfile.write(line + '\n')
        i += 1

