"""
Download sample EC classification data from UniProt
This creates a test dataset you can use immediately
"""

import requests
import pandas as pd
import time
from typing import List

def download_uniprot_ec_data(
    ec_numbers: List[str],
    max_per_ec: int = 100,
    output_file: str = 'sample_ec_data.csv'
):
    """
    Download protein sequences from UniProt for specific EC numbers
    
    Args:
        ec_numbers: List of EC numbers to download
        max_per_ec: Maximum sequences per EC number
        output_file: Output CSV filename
    """
    
    all_data = []
    
    print(f"Downloading protein data from UniProt...")
    print(f"This may take a few minutes...\n")
    
    for ec in ec_numbers:
        print(f"Downloading EC {ec}...", end=' ')
        
        # UniProt REST API
        url = 'https://rest.uniprot.org/uniprotkb/search'
        params = {
            'query': f'ec:{ec} AND reviewed:true',
            'format': 'tsv',
            'fields': 'accession,id,protein_name,sequence,ec,length',
            'size': max_per_ec
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            lines = response.text.strip().split('\n')
            
            count = 0
            for line in lines[1:]:  # Skip header
                parts = line.split('\t')
                if len(parts) >= 4:
                    accession = parts[0]
                    protein_name = parts[2] if len(parts) > 2 else ''
                    sequence = parts[3] if len(parts) > 3 else parts[1]
                    length = len(sequence)
                    
                    # Skip if sequence is too short or too long
                    if 50 <= length <= 1000:
                        all_data.append({
                            'accession': accession,
                            'protein_name': protein_name,
                            'sequence': sequence,
                            'EC': ec,
                            'length': length
                        })
                        count += 1
            
            print(f"✓ {count} sequences")
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\n{'='*60}")
    print(f"Downloaded {len(df)} total sequences")
    print(f"EC numbers: {df['EC'].nunique()}")
    print(f"\nEC distribution:")
    print(df['EC'].value_counts())
    print(f"{'='*60}\n")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    # Download a diverse set of enzyme classes
    ec_numbers = [
        # Serine proteases (EC 3.4.21.x) - well-studied, many samples
        '3.4.21.1',   # Chymotrypsin
        '3.4.21.5',   # Thrombin
        '3.4.21.4',   # Trypsin
        
        # Serine proteases (different subclass for comparison)
        '3.4.22.1',   # Papain (cysteine protease)
        
        # Oxidoreductases (EC 1.x.x.x) - different class
        '1.1.1.1',    # Alcohol dehydrogenase
        '1.1.1.27',   # Lactate dehydrogenase
        
        # Transferases (EC 2.x.x.x)
        '2.7.11.1',   # Serine/threonine protein kinase
        
        # Lyases (EC 4.x.x.x)
        '4.2.1.1',    # Carbonic anhydrase
    ]
    
    print("="*60)
    print("Downloading Sample EC Classification Dataset from UniProt")
    print("="*60)
    print()
    
    output_file = download_uniprot_ec_data(
        ec_numbers=ec_numbers,
        max_per_ec=100,
        output_file='../sample_ec_data.csv'
    )
    
    print("\nNow you can run the analysis:")
    print(f"python ec_pipeline.py --data {output_file} --min-samples 10")
