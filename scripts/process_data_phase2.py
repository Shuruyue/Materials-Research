
"""
Script: process_data_phase2.py
Purpose: Regenerate graph data with M3GNet 3-body indices.
"""
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from atlas.models.graph_builder import CrystalGraphBuilder
from atlas.config import get_config

def process_data(dataset_name="dft_3d", limit=None):
    print(f"Loading raw data from JARVIS ({dataset_name})...")
    data = jdata(dataset_name)
    
    if limit:
        data = data[:limit]
        
    print(f"Processing {len(data)} structures for M3GNet (3-body enabled)...")
    
    builder = CrystalGraphBuilder(cutoff=5.0, max_neighbors=12, compute_3body=True)
    processed_graphs = []
    
    for entry in tqdm(data):
        atoms = Atoms.from_dict(entry["atoms"])
        structure = atoms.pymatgen_converter()
        
        # Get properties
        props = {
            "formation_energy": entry.get("formation_energy_peratom", float("nan")),
            "band_gap": entry.get("optb88vdw_bandgap", float("nan")),
            "bulk_modulus": entry.get("bulk_modulus_kv", float("nan")),
            "shear_modulus": entry.get("shear_modulus_gv", float("nan")),
        }
        
        try:
            graph = builder.structure_to_pyg(structure, **props)
            # Store ID for tracking
            graph.jid = entry["jid"]
            processed_graphs.append(graph)
        except Exception as e:
            print(f"Failed to process {entry['jid']}: {e}")
            
    # Save
    save_dir = Path("data/processed")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "m3gnet_graphs.pt"
    
    print(f"Saving {len(processed_graphs)} graphs to {save_path}...")
    torch.save(processed_graphs, save_path)
    print("Done! Data optimized for Phase 2.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of structures for testing")
    args = parser.parse_args()
    
    process_data(limit=args.limit)
