
"""
Script: process_data_phase2.py
Purpose: Regenerate graph data with M3GNet 3-body indices for Phase 2.
Optimizations:
- Extracts ALL 9 available properties (Discovery Mode).
- Uses multiprocessing for fast graph construction.
- Robust error handling and caching.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.config import get_config

# Full 9-Property Matrix
PROPERTY_MAP = {
    "formation_energy_peratom": "formation_energy",
    "optb88vdw_bandgap": "band_gap",
    "mbj_bandgap": "band_gap_mbj",
    "bulk_modulus_kv": "bulk_modulus",
    "shear_modulus_gv": "shear_modulus",
    "dfpt_piezo_max_dielectric": "dielectric",
    "dfpt_piezo_max_eij": "piezoelectric",
    "spillage": "spillage",
    "ehull": "ehull",
}

def _worker(entry):
    """Picklable worker function for parallel processing."""
    try:
        atoms = Atoms.from_dict(entry["atoms"])
        structure = atoms.pymatgen_converter()
        
        # Extract all 9 properties
        props = {}
        for jarvis_col, my_name in PROPERTY_MAP.items():
            val = entry.get(jarvis_col, float("nan"))
            if val == "na" or val is None:
                val = float("nan")
            props[my_name] = float(val)
            
        # Build Graph (Expensive part)
        # 3-body=True is critical for Phase 2 E3NN models
        builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12, compute_3body=True)
        graph = builder.structure_to_pyg(structure, **props)
        graph.jid = entry["jid"]
        return graph
    except Exception as e:
        return None

def process_data(dataset_name="dft_3d", limit=None, n_workers=None):
    config = get_config()
    save_dir = config.paths.processed_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "phase2_m3gnet_graphs.pt"
    
    if save_path.exists() and limit is None:
        print(f"✅ Found existing data at {save_path}")
        print("   Delete this file if you want to force regeneration.")
        return

    print(f"Loading raw data from JARVIS ({dataset_name})...")
    data = jdata(dataset_name)
    
    if limit:
        print(f"⚠️ LIMITING TO {limit} SAMPLES FOR TESTING")
        data = data[:limit]
        
    n_workers = n_workers or max(1, multiprocessing.cpu_count() - 2)
    print(f"🚀 Processing {len(data)} structures with {n_workers} workers...")
    print(f"   extracting ALL properties: {list(PROPERTY_MAP.values())}")
    
    processed_graphs = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, entry): entry["jid"] for entry in data}
        
        for future in tqdm(as_completed(futures), total=len(data), unit="cryst"):
            res = future.result()
            if res is not None:
                processed_graphs.append(res)
            
    print(f"✨ Successfully processed {len(processed_graphs)}/{len(data)} graphs.")
    
    print(f"Saving to {save_path}...")
    torch.save(processed_graphs, save_path)
    print("Done! Phase 2 data ready.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of structures for testing")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers")
    args = parser.parse_args()
    
    # Windows-safe multiprocessing
    multiprocessing.freeze_support()
    process_data(limit=args.limit, n_workers=args.workers)
