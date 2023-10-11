import csv
import numpy as np
from typing import Dict, Any
from sidechainnet.utils.measure import get_seq_coords_and_angles
from prody import fetchPDB, parsePDB
import jsonlines
import multiprocessing as mp
import time
mp.set_start_method(method='forkserver', force=True)

def process(pdb_id, ECnumber):
    start = time.time()
    try:
        chain = parsePDB(fetchPDB(pdb_id, compressed=False))
        _, coords, seq, _, _ = get_seq_coords_and_angles(chain)
    except:
        return
    coords = coords.reshape((len(seq), 14, 3))
    coords = eval(np.array2string(coords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
    print(f"PDB file {pdb_id} is processed with {time.time() - start} seconds. ")
    return {
        "pdb": pdb_id, 
        "seq": seq, 
        "coords": coords, 
        "ECnumber": ECnumber,
    }

if __name__ == "__main__":
    start = time.time()
    csv.field_size_limit(2 ** 20)
    count = 0
    UniProtEntryList = []
    inputs = []
    with open('../data/pretrain/pdb_chain_enzyme.tsv', 'r', encoding='gbk', errors='ignore') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            count += 1
            if count > 2 and row[2] not in UniProtEntryList:
                UniProtEntryList.append(row[2])
                inputs.append((row[0], row[3]))
    
    print(f"data collected, processing with {mp.cpu_count() // 4 * 3} / {mp.cpu_count()} CPUs...... ")
    
    with mp.Pool(mp.cpu_count() // 4 * 3) as p:
        outputs = p.starmap(process, inputs)
    print("data processed, writing to ../out.jsonl ...... ")
        
    with jsonlines.open('../full.jsonl', 'w') as writer:
        for output in outputs:
            if output is not None:
                writer.write(output)
    print(f"Data processed and saved in ../out.jsonl, elapsed {time.time() - start} seconds. ")