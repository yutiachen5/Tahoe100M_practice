from datasets import load_dataset
import gzip
import pandas as pd

drug_df = pd.read_csv('/hpc/home/yc583/Tahoe100M_practice/data/drug_table.csv')
drug = set(drug_df['drug_name'])
cell_df = pd.read_csv('/hpc/home/yc583/Tahoe100M_practice/data/cell_table.csv')
cell = set(cell_df['cell_name'])

tahoe_100m_ds = load_dataset('vevotx/Tahoe-100M', streaming=True, split='train')
N = 10_000_000  
i = 0

with gzip.open('/hpc/group/naderilab/eleanor/prose_data/data/Tahoe_downsampled_diversified.txt.gz', 'wt', encoding='utf-8', compresslevel=5) as outfile:
    outfile.write('genes\texpressions\tcanonical_smiles\tdrug\tcell_line_id\n')
    for record in tahoe_100m_ds:
        if (record['drug'] in drug) and (record['cell_line_id'] in cell):
            genes_str = ','.join(map(str, record['genes']))
            expressions_str = ','.join(map(str, record['expressions']))
            line = '\t'.join([genes_str, expressions_str, record['canonical_smiles'], record['drug'], record['cell_line_id']])
            outfile.write(line + '\n')
        else:
            continue
        i += 1
        if i%N == 0:
            print(i)

