#!/usr/bin/env python3
"""
ESM-Only Data Overview
Analyze data availability and model feasibility for ESM-only QSAR modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path
from papyrus_scripts import PapyrusDataset
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ESMOnlyDataOverview:
    """
    Analyze data availability and model feasibility for ESM-only QSAR modeling.
    """
    
    def __init__(self, embeddings_path, targets_path, output_dir="analyses/qsar_papyrus_esm_only"):
        """
        Initialize the ESM-only data overview analyzer.
        
        Args:
            embeddings_path (str): Path to ESM embeddings .npy file
            targets_path (str): Path to targets CSV file
            output_dir (str): Output directory for results
        """
        self.embeddings_path = embeddings_path
        self.targets_path = targets_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load all required data."""
        logger.info("Loading data for ESM-only analysis...")
        
        # Load ESM embeddings
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            logger.info(f"Loaded ESM embeddings: {self.embeddings.shape}")
        else:
            raise FileNotFoundError(f"ESM embeddings file not found: {self.embeddings_path}")
        
        # Load targets data
        if os.path.exists(self.targets_path):
            self.targets_df = pd.read_csv(self.targets_path)
            logger.info(f"Loaded targets data: {self.targets_df.shape}")
        else:
            raise FileNotFoundError(f"Targets file not found: {self.targets_path}")
        
        # Load protein check results
        protein_check_path = "../qsar_papyrus_esm_emb/papyrus_protein_check_results.csv"
        if os.path.exists(protein_check_path):
            self.protein_check_df = pd.read_csv(protein_check_path)
            logger.info(f"Loaded protein check results: {self.protein_check_df.shape}")
        else:
            raise FileNotFoundError(f"Protein check results file not found: {protein_check_path}")
        
        # Load Papyrus dataset
        logger.info("Loading Papyrus dataset...")
        try:
            self.papyrus_dataset = PapyrusDataset()
            self.papyrus_df = self.papyrus_dataset.to_dataframe()
            logger.info(f"Loaded Papyrus dataset: {self.papyrus_df.shape}")
        except Exception as e:
            logger.error(f"Error loading Papyrus dataset: {e}")
            raise
    
    def get_esm_embedding(self, uniprot_id):
        """Get ESM embedding for a specific UniProt ID."""
        protein_mask = self.targets_df['name2_entry'] == uniprot_id
        if protein_mask.any():
            protein_idx = self.targets_df[protein_mask].index[0]
            return self.embeddings[protein_idx]
        else:
            return None
    
    def get_protein_activities(self, protein_name):
        """Get bioactivity data for a specific protein."""
        protein_activities = self.papyrus_df[
            (self.papyrus_df['protein_name'] == protein_name) |
            (self.papyrus_df['protein_name'].str.contains(protein_name, case=False, na=False))
        ]
        
        if len(protein_activities) == 0:
            return pd.DataFrame()
        
        # Clean the data
        protein_activities = protein_activities.dropna(subset=['pchembl_value_Mean'])
        protein_activities = protein_activities[protein_activities['pchembl_value_Mean'].notna()]
        
        return protein_activities
    
    def test_model_feasibility(self, protein_name, uniprot_id):
        """
        Test if a protein is feasible for ESM-only QSAR modeling.
        
        Args:
            protein_name (str): Protein name
            uniprot_id (str): UniProt ID
            
        Returns:
            dict: Feasibility test results
        """
        # Check ESM embedding availability
        esm_embedding = self.get_esm_embedding(uniprot_id)
        esm_available = esm_embedding is not None
        
        # Get protein activities
        protein_activities = self.get_protein_activities(protein_name)
        total_activities = len(protein_activities)
        
        # Check if we have sufficient data
        sufficient_activities = total_activities >= 10
        
        # Check if target variable is available and valid
        valid_targets = False
        if total_activities > 0:
            target_values = protein_activities['pchembl_value_Mean'].values
            valid_targets = np.all(~(np.isnan(target_values) | np.isinf(target_values)))
        
        # Determine model status
        if not esm_available:
            model_status = 'no_esm'
        elif total_activities == 0:
            model_status = 'no_activities'
        elif not sufficient_activities:
            model_status = 'insufficient_data'
        elif not valid_targets:
            model_status = 'invalid_targets'
        else:
            model_status = 'success'
        
        return {
            'protein_name': protein_name,
            'uniprot_id': uniprot_id,
            'total_activities': total_activities,
            'esm_available': esm_available,
            'sufficient_activities': sufficient_activities,
            'valid_targets': valid_targets,
            'model_status': model_status
        }
    
    def analyze_all_proteins(self):
        """
        Analyze all proteins for ESM-only QSAR modeling feasibility.
        
        Returns:
            pd.DataFrame: Analysis results for all proteins
        """
        logger.info("Analyzing all proteins for ESM-only QSAR modeling...")
        
        results = []
        
        for _, row in self.protein_check_df.iterrows():
            protein_name = row['protein_name']
            uniprot_id = protein_name  # The name2_entry is the UniProt ID
            
            result = self.test_model_feasibility(protein_name, uniprot_id)
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = self.output_dir / "esm_only_data_overview_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved analysis results to {results_path}")
        
        return results_df
    
    def create_overview_plots(self, results_df):
        """Create overview plots for the ESM-only data analysis."""
        logger.info("Creating overview plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ESM-Only QSAR Data Overview', fontsize=16, fontweight='bold')
        
        # 1. Activities per protein
        activities_sorted = results_df.sort_values('total_activities', ascending=False)
        bars1 = axes[0, 0].bar(range(len(activities_sorted)), activities_sorted['total_activities'], 
                                color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Activities per Protein', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Proteins')
        axes[0, 0].set_ylabel('Number of Activities')
        
        # Add value labels on top bars
        for i, bar in enumerate(bars1[:10]):  # Only label top 10
            if activities_sorted.iloc[i]['total_activities'] > 0:
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                               str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
        
        # 2. ESM availability
        esm_counts = results_df['esm_available'].value_counts()
        colors = ['lightcoral', 'lightgreen']
        wedges, texts, autotexts = axes[0, 1].pie(esm_counts.values, labels=esm_counts.index, 
                                                   colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('ESM Embedding Availability', fontsize=12, fontweight='bold')
        
        # 3. Model status distribution
        model_status_counts = results_df['model_status'].value_counts()
        bars3 = axes[0, 2].bar(range(len(model_status_counts)), model_status_counts.values, 
                                color='lightblue', alpha=0.7)
        axes[0, 2].set_title('Model Status Distribution', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Status')
        axes[0, 2].set_ylabel('Number of Proteins')
        axes[0, 2].set_xticks(range(len(model_status_counts)))
        axes[0, 2].set_xticklabels(model_status_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars3):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(int(bar.get_height())), ha='center', va='bottom', fontsize=8)
        
        # 4. Activities vs model success
        successful = results_df[results_df['model_status'] == 'success']
        unsuccessful = results_df[results_df['model_status'] != 'success']
        
        axes[1, 0].scatter(successful['total_activities'], [1]*len(successful), 
                           color='green', alpha=0.6, s=50, label='Successful')
        axes[1, 0].scatter(unsuccessful['total_activities'], [0]*len(unsuccessful), 
                           color='red', alpha=0.6, s=50, label='Unsuccessful')
        axes[1, 0].set_title('Activities vs Model Success', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Activities')
        axes[1, 0].set_ylabel('Model Success')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Activities distribution
        axes[1, 1].hist(results_df['total_activities'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_title('Activities Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Activities')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        
        total_proteins = len(results_df)
        proteins_with_activities = len(results_df[results_df['total_activities'] > 0])
        proteins_with_esm = len(results_df[results_df['esm_available'] == True])
        successful_models = len(results_df[results_df['model_status'] == 'success'])
        
        summary_text = f"""
        ESM-ONLY DATASET OVERVIEW
        
        Total Proteins: {total_proteins}
        Proteins with Activities: {proteins_with_activities} ({proteins_with_activities/total_proteins*100:.1f}%)
        Proteins with ESM: {proteins_with_esm} ({proteins_with_esm/total_proteins*100:.1f}%)
        Successful Models: {successful_models} ({successful_models/total_proteins*100:.1f}%)
        
        ACTIVITY STATISTICS
        Mean Activities: {results_df['total_activities'].mean():.1f}
        Median Activities: {results_df['total_activities'].median():.1f}
        Max Activities: {results_df['total_activities'].max()}
        Min Activities: {results_df['total_activities'].min()}
        
        TOP 5 PROTEINS BY ACTIVITIES
        """
        
        top_proteins = results_df.nlargest(5, 'total_activities')
        for idx, row in top_proteins.iterrows():
            status_symbol = "✅" if row['model_status'] == 'success' else "❌"
            summary_text += f"{status_symbol} {row['protein_name']}: {row['total_activities']} activities\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "esm_only_data_overview.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved overview plot to {plot_path}")
        
        plt.show()
    
    def create_detailed_report(self, results_df):
        """Create a detailed text report of the analysis."""
        logger.info("Creating detailed report...")
        
        total_proteins = len(results_df)
        proteins_with_activities = len(results_df[results_df['total_activities'] > 0])
        proteins_with_esm = len(results_df[results_df['esm_available'] == True])
        successful_models = len(results_df[results_df['model_status'] == 'success'])
        
        report = f"""
{'='*60}
ESM-ONLY QSAR DATA OVERVIEW REPORT
{'='*60}

DATASET SUMMARY:
- Total proteins analyzed: {total_proteins}
- Proteins with bioactivity data: {proteins_with_activities} ({proteins_with_activities/total_proteins*100:.1f}%)
- Proteins with ESM embeddings: {proteins_with_esm} ({proteins_with_esm/total_proteins*100:.1f}%)
- Proteins ready for modeling: {successful_models} ({successful_models/total_proteins*100:.1f}%)

ACTIVITY STATISTICS:
- Mean activities per protein: {results_df['total_activities'].mean():.1f}
- Median activities per protein: {results_df['total_activities'].median():.1f}
- Maximum activities: {results_df['total_activities'].max()}
- Minimum activities: {results_df['total_activities'].min()}

MODEL STATUS BREAKDOWN:
"""
        
        status_counts = results_df['model_status'].value_counts()
        for status, count in status_counts.items():
            percentage = (count / total_proteins) * 100
            report += f"- {status}: {count} proteins ({percentage:.1f}%)\n"
        
        report += f"""
TOP 10 PROTEINS BY ACTIVITY COUNT:
"""
        
        top_proteins = results_df.nlargest(10, 'total_activities')
        for idx, row in top_proteins.iterrows():
            status_symbol = "✅" if row['model_status'] == 'success' else "❌"
            report += f"{status_symbol} {row['protein_name']}: {row['total_activities']} activities\n"
        
        report += f"""
PROTEINS READY FOR ESM-ONLY MODELING:
"""
        
        ready_proteins = results_df[results_df['model_status'] == 'success']
        if len(ready_proteins) > 0:
            for idx, row in ready_proteins.iterrows():
                report += f"✅ {row['protein_name']}: {row['total_activities']} activities\n"
        else:
            report += "No proteins are currently ready for ESM-only modeling.\n"
        
        report += f"""
{'='*60}
"""
        
        # Save report
        report_path = self.output_dir / "esm_only_data_overview_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved detailed report to {report_path}")
        
        # Print report to console
        print(report)
        
        return report


def main():
    """Main function to run ESM-only data overview analysis."""
    # File paths
    embeddings_path = "../../embeddings.npy"
    targets_path = "../qsar_papyrus_esm_emb/targets_w_sequences.csv"
    
    # Initialize analyzer
    analyzer = ESMOnlyDataOverview(embeddings_path, targets_path)
    
    # Analyze all proteins
    results_df = analyzer.analyze_all_proteins()
    
    # Create overview plots
    analyzer.create_overview_plots(results_df)
    
    # Create detailed report
    analyzer.create_detailed_report(results_df)


if __name__ == "__main__":
    main() 