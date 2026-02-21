#!/usr/bin/env python3
"""
Merge multiple result files from parallel processing into a single file.
Usage: python merge_results.py
"""

import json
import glob
import os

def merge_results():
    # Find all result files
    result_files = sorted(glob.glob("ivl_results_*.json"))
    
    if not result_files:
        print("No result files found matching pattern 'ivl_results_*.json'")
        return
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f}")
    
    all_results = []
    
    for result_file in result_files:
        print(f"\nReading {result_file}...")
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results.extend(data)
                print(f"  Loaded {len(data)} samples")
        except Exception as e:
            print(f"  Error reading {result_file}: {e}")
    
    # Sort by sample_id to ensure correct order
    all_results.sort(key=lambda x: x['sample_id'])
    
    # Save merged results
    output_file = "ivl_results_merged.json"
    print(f"\nSaving merged results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Done! Merged {len(all_results)} samples into {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total samples: {len(all_results)}")
    if all_results:
        print(f"  Sample ID range: {all_results[0]['sample_id']} - {all_results[-1]['sample_id']}")

if __name__ == "__main__":
    merge_results()

