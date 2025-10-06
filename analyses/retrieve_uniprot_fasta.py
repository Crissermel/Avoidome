#!/usr/bin/env python3
"""
UniProt FASTA Retrieval Script
Retrieves FASTA files from UniProt IDs and saves them in primary_data/uniprot_fasta/

This script:
1. Reads the multi_organism_results.csv file to extract UniProt IDs
2. Downloads FASTA files for each unique UniProt ID
3. Saves files with UniProt ID as filename in primary_data/uniprot_fasta/
4. Handles errors gracefully and provides progress updates
"""

import pandas as pd
import requests
import os
from pathlib import Path
import time
from typing import List, Set
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('uniprot_fasta_retrieval.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UniProtFastaRetriever:
    """Class to handle UniProt FASTA file retrieval"""
    
    def __init__(self, output_dir: str = "primary_data/uniprot_fasta"):
        """
        Initialize the retriever
        
        Args:
            output_dir: Directory to save FASTA files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://rest.uniprot.org/uniprotkb/{}.fasta"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; UniProt-FASTA-Retriever/1.0)'
        })
    
    def extract_uniprot_ids(self, csv_path: str) -> Set[str]:
        """
        Extract all unique UniProt IDs from the multi_organism_results.csv file
        
        Args:
            csv_path: Path to the CSV file containing UniProt IDs
            
        Returns:
            Set of unique UniProt IDs
        """
        logger.info(f"Reading UniProt IDs from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            uniprot_ids = set()
            
            # Extract IDs from human_id, mouse_id, and rat_id columns
            for column in ['human_id', 'mouse_id', 'rat_id']:
                if column in df.columns:
                    # Filter out empty strings and NaN values
                    ids = df[column].dropna().astype(str)
                    ids = ids[ids != 'nan'].tolist()
                    uniprot_ids.update(ids)
            
            logger.info(f"Found {len(uniprot_ids)} unique UniProt IDs")
            return uniprot_ids
            
        except FileNotFoundError:
            logger.error(f"File not found: {csv_path}")
            return set()
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return set()
    
    def download_fasta(self, uniprot_id: str) -> bool:
        """
        Download FASTA file for a single UniProt ID
        
        Args:
            uniprot_id: UniProt ID to download
            
        Returns:
            True if successful, False otherwise
        """
        if not uniprot_id or uniprot_id == 'nan':
            return False
            
        output_file = self.output_dir / f"{uniprot_id}.fasta"
        
        # Skip if file already exists
        if output_file.exists():
            logger.info(f"File already exists: {output_file}")
            return True
        
        try:
            url = self.base_url.format(uniprot_id)
            logger.info(f"Downloading {uniprot_id} from {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if response contains FASTA data
            if response.text.startswith('>'):
                with open(output_file, 'w') as f:
                    f.write(response.text)
                logger.info(f"Successfully downloaded {uniprot_id}")
                return True
            else:
                logger.warning(f"No FASTA data received for {uniprot_id}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {uniprot_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {uniprot_id}: {e}")
            return False
    
    def download_all_fasta(self, uniprot_ids: Set[str], delay: float = 1.0) -> dict:
        """
        Download FASTA files for all UniProt IDs
        
        Args:
            uniprot_ids: Set of UniProt IDs to download
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary with download statistics
        """
        logger.info(f"Starting download of {len(uniprot_ids)} FASTA files")
        
        successful = 0
        failed = 0
        skipped = 0
        
        for i, uniprot_id in enumerate(sorted(uniprot_ids), 1):
            logger.info(f"Processing {i}/{len(uniprot_ids)}: {uniprot_id}")
            
            if self.download_fasta(uniprot_id):
                successful += 1
            else:
                failed += 1
            
            # Add delay to be respectful to UniProt servers
            if i < len(uniprot_ids):
                time.sleep(delay)
        
        # Count skipped (already existing) files
        for uniprot_id in uniprot_ids:
            if (self.output_dir / f"{uniprot_id}.fasta").exists():
                skipped += 1
        
        stats = {
            'total_requested': len(uniprot_ids),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'output_directory': str(self.output_dir)
        }
        
        logger.info(f"Download completed. Stats: {stats}")
        return stats
    
    def get_download_stats(self) -> dict:
        """
        Get statistics about downloaded files
        
        Returns:
            Dictionary with file statistics
        """
        if not self.output_dir.exists():
            return {'total_files': 0, 'file_size_mb': 0}
        
        fasta_files = list(self.output_dir.glob("*.fasta"))
        total_size = sum(f.stat().st_size for f in fasta_files)
        
        return {
            'total_files': len(fasta_files),
            'file_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in fasta_files]
        }

def main():
    """Main function to run the FASTA retrieval"""
    logger.info("Starting UniProt FASTA retrieval")
    
    # Initialize retriever
    retriever = UniProtFastaRetriever()
    
    # Path to the CSV file
    csv_path = "analyses/qsar_papyrus_modelling/multi_organism_results.csv"
    
    # Extract UniProt IDs
    uniprot_ids = retriever.extract_uniprot_ids(csv_path)
    
    if not uniprot_ids:
        logger.error("No UniProt IDs found. Exiting.")
        return
    
    # Display found IDs
    logger.info("Found UniProt IDs:")
    for uniprot_id in sorted(uniprot_ids):
        logger.info(f"  - {uniprot_id}")
    
    # Download FASTA files
    stats = retriever.download_all_fasta(uniprot_ids, delay=1.0)
    
    # Display final statistics
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total UniProt IDs: {stats['total_requested']}")
    logger.info(f"Successfully downloaded: {stats['successful']}")
    logger.info(f"Failed downloads: {stats['failed']}")
    logger.info(f"Skipped (already existed): {stats['skipped']}")
    logger.info(f"Output directory: {stats['output_directory']}")
    
    # Get file statistics
    file_stats = retriever.get_download_stats()
    logger.info(f"Total FASTA files: {file_stats['total_files']}")
    logger.info(f"Total size: {file_stats['file_size_mb']:.2f} MB")
    
    logger.info("=" * 50)
    logger.info("FASTA retrieval completed!")

if __name__ == "__main__":
    main() 