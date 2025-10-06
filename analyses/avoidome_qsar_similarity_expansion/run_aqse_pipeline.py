#!/usr/bin/env python3
"""
AQSE Pipeline Runner

Main script to run the complete Avoidome QSAR Similarity Expansion (AQSE) pipeline.
This script orchestrates all three steps: input preparation, protein similarity search, and data collection.

Author: AQSE Pipeline
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
    
    # Load input_preparation module
    spec1 = importlib.util.spec_from_file_location("input_preparation", current_dir / "01_input_preparation.py")
    input_prep_module = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(input_prep_module)
    AvoidomeInputPreparation = input_prep_module.AvoidomeInputPreparation
    
    # Load protein_similarity_search module (Papyrus-based)
    spec2 = importlib.util.spec_from_file_location("protein_similarity_search_papyrus", current_dir / "02_protein_similarity_search_papyrus.py")
    similarity_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(similarity_module)
    ProteinSimilaritySearch = similarity_module.ProteinSimilaritySearchPapyrus
    
    # Load data_collection_strategy module (Papyrus-based)
    spec3 = importlib.util.spec_from_file_location("data_collection_strategy_papyrus", current_dir / "03_data_collection_strategy_papyrus.py")
    data_collection_module = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(data_collection_module)
    DataCollectionStrategy = data_collection_module.DataCollectionStrategy
    
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
        logging.FileHandler('aqse_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AQSEPipeline:
    """Main pipeline class for AQSE workflow"""
    
    def __init__(self, config: dict):
        """
        Initialize the AQSE pipeline
        
        Args:
            config: Configuration dictionary with paths and parameters
        """
        self.config = config
        self.base_dir = Path(config['base_dir'])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.log_file = self.base_dir / "aqse_pipeline.log"
        
    def run_step1_input_preparation(self) -> dict:
        """Run Step 1: Input Preparation"""
        logger.info("="*60)
        logger.info("STEP 1: INPUT PREPARATION")
        logger.info("="*60)
        
        try:
            preparer = AvoidomeInputPreparation(
                avoidome_file=self.config['avoidome_file'],
                output_dir=str(self.base_dir / "01_input_preparation")
            )
            
            results = preparer.run_preparation()
            logger.info("Step 1 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            raise
    
    def run_step2_similarity_search(self, step1_results: dict) -> dict:
        """Run Step 2: Protein Similarity Search using Papyrus-scripts"""
        logger.info("="*60)
        logger.info("STEP 2: PROTEIN SIMILARITY SEARCH (PAPYRUS-SCRIPTS)")
        logger.info("="*60)
        
        try:
            # Import the Papyrus-based similarity search module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "protein_similarity_search_papyrus", 
                str(self.base_dir / "02_protein_similarity_search_papyrus.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            searcher = ProteinSimilaritySearch(
                input_dir=str(self.base_dir / "01_input_preparation"),
                output_dir=str(self.base_dir / "02_similarity_search"),
                papyrus_version=self.config.get('papyrus_version', '05.7')
            )
            
            results = searcher.run_similarity_search()
            logger.info("Step 2 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            raise
    
    def run_step3_data_collection(self, step2_results: dict) -> dict:
        """Run Step 3: Data Collection Strategy using Papyrus-scripts"""
        logger.info("="*60)
        logger.info("STEP 3: DATA COLLECTION STRATEGY (PAPYRUS-SCRIPTS)")
        logger.info("="*60)
        
        try:
            collector = DataCollectionStrategy(
                similarity_results_dir=str(self.base_dir / "02_similarity_search"),
                output_dir=str(self.base_dir / "03_data_collection")
            )
            
            results = collector.run_data_collection()
            logger.info("Step 3 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            raise
    
    def run_step4_qsar_models(self, step3_results: dict) -> dict:
        """Run Step 4: QSAR Model Creation"""
        logger.info("="*60)
        logger.info("STEP 4: QSAR MODEL CREATION")
        logger.info("="*60)
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "qsar_model_creation_2",
                str(self.base_dir / "04_2_qsar_model_creation.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            qsar_creator = module.AQSEQSARModelCreation(
                output_dir=str(self.base_dir / "04_qsar_models_temp"),
                avoidome_file=self.config.get('avoidome_file'),
                similarity_file=str(self.base_dir / "02_similarity_search" / "similarity_search_summary.csv")
            )
            
            results = qsar_creator.run_aqse_workflow()
            logger.info("Step 4 completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            raise
    
    def create_pipeline_summary(self, step1_results: dict, step2_results: dict, step3_results: dict, step4_results: dict = None) -> str:
        """Create a summary of the entire pipeline run"""
        logger.info("Creating pipeline summary")
        
        summary_file = self.base_dir / "pipeline_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("AQSE Pipeline Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Base Directory: {self.base_dir}\n\n")
            
            # Step 1 summary
            f.write("STEP 1: INPUT PREPARATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Sequences CSV: {step1_results.get('sequences_csv', 'N/A')}\n")
            f.write(f"BLAST Config: {step1_results.get('blast_config', 'N/A')}\n")
            f.write(f"FASTA files: {len(step1_results.get('fasta_files', []))}\n")
            f.write(f"Failed sequences: {len(step1_results.get('failed_ids', []))}\n\n")
            
            # Step 2 summary
            f.write("STEP 2: PROTEIN SIMILARITY SEARCH\n")
            f.write("-" * 30 + "\n")
            f.write(f"BLAST files processed: {len(step2_results.get('blast_files', []))}\n")
            f.write(f"Similarity matrices: {len(step2_results.get('matrices', {}))}\n")
            f.write(f"Summary statistics: {len(step2_results.get('summary_df', []))} rows\n\n")
            
            # Step 3 summary
            f.write("STEP 3: DATA COLLECTION STRATEGY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Expanded datasets: {len(step3_results.get('expanded_datasets', {}))}\n")
            f.write(f"Statistics: {len(step3_results.get('statistics', []))} thresholds\n\n")
            
            # Step 4 summary
            if step4_results:
                f.write("STEP 4: QSAR MODEL CREATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Models created: {step4_results.get('models_created', 0)}\n")
                f.write(f"Proteins skipped: {step4_results.get('proteins_skipped', 0)}\n")
                f.write(f"Total proteins: {step4_results.get('total_proteins', 0)}\n")
                f.write(f"Morgan fingerprints: {step4_results.get('step_0_results', {}).get('valid_fingerprints', 0)} compounds\n")
                f.write(f"Results exported to CSV and JSON\n\n")
            
            # Dataset details
            if 'expanded_datasets' in step3_results:
                f.write("DATASET DETAILS\n")
                f.write("-" * 30 + "\n")
                for threshold, df in step3_results['expanded_datasets'].items():
                    f.write(f"{threshold}: {len(df)} data points\n")
        
        logger.info(f"Pipeline summary saved to: {summary_file}")
        return str(summary_file)
    
    def run_pipeline(self) -> dict:
        """Run the complete AQSE pipeline"""
        logger.info("Starting AQSE Pipeline")
        logger.info(f"Base directory: {self.base_dir}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Input Preparation
            step1_results = self.run_step1_input_preparation()
            
            # Step 2: Protein Similarity Search
            step2_results = self.run_step2_similarity_search(step1_results)
            
            # Step 3: Data Collection Strategy
            step3_results = self.run_step3_data_collection(step2_results)
            
            # Step 4: QSAR Model Creation
            step4_results = self.run_step4_qsar_models(step3_results)
            
            # Create summary
            summary_file = self.create_pipeline_summary(step1_results, step2_results, step3_results, step4_results)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("AQSE PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total runtime: {duration}")
            logger.info(f"Summary file: {summary_file}")
            logger.info("="*60)
            
            return {
                'step1_results': step1_results,
                'step2_results': step2_results,
                'step3_results': step3_results,
                'step4_results': step4_results,
                'summary_file': summary_file,
                'duration': str(duration)
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def create_default_config():
    """Create default configuration"""
    return {
        'base_dir': '/home/serramelendezcsm/RA/Avoidome/analyses/avoidome_qsar_similarity_expansion',
        'avoidome_file': '/home/serramelendezcsm/RA/Avoidome/processed_data/papyrus_protein_check_results.csv',
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
    parser = argparse.ArgumentParser(description='Run AQSE Pipeline with Papyrus-scripts')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON)')
    parser.add_argument('--papyrus-version', type=str, default='05.7', help='Papyrus dataset version (default: 05.7)')
    parser.add_argument('--base-dir', type=str, help='Base directory for output')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], help='Run only specific step')
    
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
    pipeline = AQSEPipeline(config)
    
    try:
        if args.step:
            # Run specific step
            if args.step == 1:
                results = pipeline.run_step1_input_preparation()
            elif args.step == 2:
                step1_results = {}  # Would need to load from previous run
                results = pipeline.run_step2_similarity_search(step1_results)
            elif args.step == 3:
                step2_results = {}  # Would need to load from previous run
                results = pipeline.run_step3_data_collection(step2_results)
            elif args.step == 4:
                step3_results = {}  # Would need to load from previous run
                results = pipeline.run_step4_qsar_models(step3_results)
        else:
            # Run complete pipeline
            results = pipeline.run_pipeline()
        
        print("\n" + "="*60)
        print("AQSE PIPELINE COMPLETED")
        print("="*60)
        print(f"Results saved to: {config['base_dir']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()