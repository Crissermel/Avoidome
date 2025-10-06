#!/usr/bin/env python3
"""
Run Classification Pipeline

This script provides a simple interface to run the classification pipeline
with different options and configurations.

Date: 2025-01-30
"""

import argparse
import logging
import sys
from pathlib import Path

from minimal_papyrus_esm_classification import PapyrusESMQSARClassifier
from analyze_classification_results import ClassificationResultsAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/serramelendezcsm/RA/Avoidome/analyses/qsar_papyrus_esm_emb_classification/run_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_classification(threshold: float = 6.0):
    """Run classification with specified threshold"""
    logger.info(f"Running classification with threshold: {threshold}")
    
    model = PapyrusESMQSARClassifier(activity_threshold=threshold)
    model.run_classification_pipeline()
    model.save_results()
    
    logger.info("Classification completed!")

def run_full_pipeline(threshold: float = 6.0):
    """Run the complete pipeline: classification and analysis"""
    logger.info("Running complete classification pipeline...")
    
    # Step 1: Run classification
    logger.info("Step 1: Running classification")
    run_classification(threshold)
    
    # Step 2: Analyze results
    logger.info("Step 2: Analyzing results")
    analyzer = ClassificationResultsAnalyzer()
    analyzer.analyze_results()
    
    logger.info("Complete pipeline finished!")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run QSAR classification pipeline')
    parser.add_argument('--mode', choices=['classify', 'analyze', 'full', 'test'], 
                       default='classify', help='Pipeline mode')
    parser.add_argument('--threshold', type=float, default=6.0,
                       help='Activity threshold for classification')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("QSAR CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    
    try:
        if args.mode == 'test':
            logger.info("Running in test mode...")
            from test_classification import main as test_main
            test_main()
        elif args.mode == 'classify':
            run_classification(args.threshold)
        elif args.mode == 'analyze':
            analyzer = ClassificationResultsAnalyzer()
            analyzer.analyze_results()
        elif args.mode == 'full':
            run_full_pipeline(args.threshold)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 