#!/usr/bin/env python3
"""
Debug script to test full similarity search for mouse and rat
"""

import sys
from pathlib import Path
import logging

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mouse_similarity():
    """Test similarity search for mouse only"""
    
    # Import the similarity search class
    import importlib.util
    spec = importlib.util.spec_from_file_location("organism_similarity_search", current_dir / "02_organism_similarity_search.py")
    similarity_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(similarity_module)
    OrganismSimilaritySearch = similarity_module.OrganismSimilaritySearch
    
    # Initialize
    input_dir = str(current_dir / "01_organism_mapping")
    output_dir = str(current_dir / "debug_similarity")
    searcher = OrganismSimilaritySearch(input_dir, output_dir, papyrus_version='05.7')
    
    # Test mouse similarity search
    logger.info("Testing mouse similarity search...")
    try:
        results = searcher.run_organism_similarity_search('mouse')
        if results:
            logger.info(f"Mouse similarity search completed: {len(results.get('organism_sequences', {}))} sequences processed")
        else:
            logger.warning("Mouse similarity search returned no results")
    except Exception as e:
        logger.error(f"Error in mouse similarity search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mouse_similarity()





