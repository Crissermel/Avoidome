#!/usr/bin/env python3
"""
AQSE-org Pipeline Runner

Main script to run the complete Avoidome QSAR Similarity Expansion (AQSE-org) pipeline.
This script orchestrates all steps: organism-specific protein mapping, similarity search, and QSAR model creation.

Author: AQSE-org Pipeline
Date: 2025
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import modules with proper path handling
try:
    # Import from the numbered files
    import importlib.util
    
    # Load organism_protein_mapping module
    spec1 = importlib.util.spec_from_file_location("organism_protein_mapping", current_dir / "01_organism_protein_mapping.py")
    mapping_module = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(mapping_module)
    OrganismProteinMapping = mapping_module.OrganismProteinMapping
    
    # Load organism_similarity_search module
    spec2 = importlib.util.spec_from_file_location("organism_similarity_search", current_dir / "02_organism_similarity_search.py")
    similarity_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(similarity_module)
    OrganismSimilaritySearch = similarity_module.OrganismSimilaritySearch
    
    # Load organism_qsar_model_creation module
    spec3 = importlib.util.spec_from_file_location("organism_qsar_model_creation", current_dir / "03_organism_qsar_model_creation.py")
    qsar_module = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(qsar_module)
    OrganismQSARModelCreation = qsar_module.OrganismQSARModelCreation
    
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Files in directory: {list(current_dir.glob('*.py'))}")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aqse_org_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AQSEOrgPipeline:
    """Main pipeline class for AQSE-org workflow"""
    
    def __init__(self, config: dict):
        """
        Initialize the AQSE-org pipeline
        
        Args:
            config: Configuration dictionary with paths and parameters
        """
        self.config = config
        self.base_dir = Path(config['base_dir'])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.log_file = self.base_dir / "aqse_org_pipeline.log"
        
    def run_step1_organism_mapping(self) -> dict:
        """Run Step 1: Organism-Specific Protein Mapping"""
        logger.info("="*60)
        logger.info("STEP 1: ORGANISM-SPECIFIC PROTEIN MAPPING")
        logger.info("="*60)
        
        try:
            mapper = OrganismProteinMapping(
                avoidome_file=self.config['avoidome_file'],
                organism_mapping_file=self.config['organism_mapping_file'],
                output_dir=str(self.base_dir / "01_organism_mapping")
            )
            
            results = mapper.run_organism_mapping()
            logger.info("Step 1 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            raise
    
    def run_step2_similarity_search(self, step1_results: dict) -> dict:
        """Run Step 2: Organism-Specific Protein Similarity Search"""
        logger.info("="*60)
        logger.info("STEP 2: ORGANISM-SPECIFIC PROTEIN SIMILARITY SEARCH")
        logger.info("="*60)
        
        try:
            searcher = OrganismSimilaritySearch(
                input_dir=str(self.base_dir / "01_organism_mapping"),
                output_dir=str(self.base_dir / "02_similarity_search"),
                papyrus_version=self.config.get('papyrus_version', '05.7')
            )
            
            results = searcher.run_all_organism_similarity_searches()
            logger.info("Step 2 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            raise
    
    def run_step3_qsar_models(self, step2_results: dict) -> dict:
        """Run Step 3: Organism-Specific QSAR Model Creation"""
        logger.info("="*60)
        logger.info("STEP 3: ORGANISM-SPECIFIC QSAR MODEL CREATION")
        logger.info("="*60)
        
        try:
            qsar_creator = OrganismQSARModelCreation(
                output_dir=str(self.base_dir / "03_qsar_models"),
                organism_mapping_file=str(self.base_dir / "01_organism_mapping" / "organism_mappings_summary.csv"),
                similarity_file=str(self.base_dir / "02_similarity_search" / "all_organisms_similarity_summary.csv")
            )
            
            results = qsar_creator.run_organism_qsar_workflow()
            logger.info("Step 3 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            raise
    
    def create_pipeline_summary(self, step1_results: dict, step2_results: dict, step3_results: dict = None) -> str:
        """Create a summary of the entire pipeline run"""
        logger.info("Creating pipeline summary")
        
        summary_file = self.base_dir / "aqse_org_pipeline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AQSE-org Pipeline Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Directory: {self.base_dir}\n\n")
            
            # Step 1 summary
            f.write("STEP 1: ORGANISM-SPECIFIC PROTEIN MAPPING\n")
            f.write("-" * 30 + "\n")
            f.write(f"Organisms processed: human, mouse, rat\n")
            f.write(f"Mapping summary: {step1_results.get('mappings_summary', 'N/A')}\n")
            f.write(f"Sequence files: {len(step1_results.get('sequence_files', {}))}\n")
            f.write(f"FASTA files: {sum(len(files) for files in step1_results.get('fasta_files', {}).values())}\n\n")
            
            # Step 2 summary
            f.write("STEP 2: ORGANISM-SPECIFIC SIMILARITY SEARCH\n")
            f.write("-" * 30 + "\n")
            f.write(f"Organisms processed: {len(step2_results)}\n")
            for organism, org_results in step2_results.items():
                f.write(f"{organism.capitalize()}: {len(org_results.get('organism_sequences', {}))} proteins processed\n")
            f.write(f"Similarity matrices: {sum(len(org_results.get('matrices', {})) for org_results in step2_results.values())}\n\n")
            
            # Step 3 summary
            if step3_results:
                f.write("STEP 3: ORGANISM-SPECIFIC QSAR MODEL CREATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Models created: {step3_results.get('models_created', 0)}\n")
                f.write(f"Proteins skipped: {step3_results.get('proteins_skipped', 0)}\n")
                f.write(f"Total organisms: {step3_results.get('total_organisms', 0)}\n")
                f.write(f"Morgan fingerprints: {step3_results.get('step_0_results', {}).get('valid_fingerprints', 0)} compounds\n")
                f.write(f"Results exported to CSV\n\n")
            
            # Organism details
            f.write("ORGANISM DETAILS\n")
            f.write("-" * 30 + "\n")
            for organism in ['human', 'mouse', 'rat']:
                if organism in step1_results.get('organism_mappings', {}):
                    f.write(f"{organism.capitalize()}: {len(step1_results['organism_mappings'][organism])} proteins mapped\n")
        
        logger.info(f"Pipeline summary saved to: {summary_file}")
        return str(summary_file)
    
    def run_pipeline(self) -> dict:
        """Run the complete AQSE-org pipeline"""
        logger.info("Starting AQSE-org Pipeline")
        logger.info(f"Base directory: {self.base_dir}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Organism-Specific Protein Mapping
            step1_results = self.run_step1_organism_mapping()
            
            # Step 2: Organism-Specific Protein Similarity Search
            step2_results = self.run_step2_similarity_search(step1_results)
            
            # Step 3: Organism-Specific QSAR Model Creation
            step3_results = self.run_step3_qsar_models(step2_results)
            
            # Create summary
            summary_file = self.create_pipeline_summary(step1_results, step2_results, step3_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("AQSE-ORG PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total runtime: {duration}")
            logger.info(f"Summary file: {summary_file}")
            logger.info("="*60)
            
            return {
                'step1_results': step1_results,
                'step2_results': step2_results,
                'step3_results': step3_results,
                'summary_file': summary_file,
                'duration': str(duration)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def create_default_config():
    """Create default configuration"""
    return {
        'base_dir': '/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion_organism',
        'avoidome_file': '/home/serramelendezcsm/RA/Avoidome/primary_data/avoidome_prot_list.csv',
        'organism_mapping_file': '/home/serramelendezcsm/RA/Avoidome/primary_data/prot_orgs_extended.csv',
        'papyrus_version': '05.7',  # Papyrus dataset version
        'similarity_thresholds': {
            'high': 70.0,
            'medium': 50.0,
            'low': 30.0
        },
        'activity_thresholds': {
            'high_activity': 1.0,
            'medium_activity': 0.0,
            'low_activity': -1.0
        }
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run AQSE-org Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--papyrus-version', type=str, default='05.7', help='Papyrus dataset version (default: 05.7)')
    parser.add_argument('--base-dir', type=str, help='Base directory for output')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], help='Run only specific step')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Override with command line arguments
    if args.papyrus_version:
        config['papyrus_version'] = args.papyrus_version
    if args.base_dir:
        config['base_dir'] = args.base_dir
    
    # Initialize pipeline
    pipeline = AQSEOrgPipeline(config)
    
    try:
        if args.step:
            # Run specific step
            if args.step == 1:
                results = pipeline.run_step1_organism_mapping()
            elif args.step == 2:
                step1_results = {}  # Would need to load from previous run
                results = pipeline.run_step2_similarity_search(step1_results)
            elif args.step == 3:
                step2_results = {}  # Would need to load from previous run
                results = pipeline.run_step3_qsar_models(step2_results)
        else:
            # Run complete pipeline
            results = pipeline.run_pipeline()
        
        print("\n" + "="*60)
        print("AQSE-ORG PIPELINE COMPLETED")
        print("="*60)
        print(f"Results saved to: {config['base_dir']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
