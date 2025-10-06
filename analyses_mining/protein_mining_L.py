import sys
import re
import time
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import json

import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProteinToxicityResult:
    """Data structure for storing protein-toxicity associations"""

    pmid: str
    title: str
    abstract: str
    proteins: List[str]
    toxicity_terms: List[str]
    confidence_score: float
    extraction_method: str
    publication_date: str


class BasePubMedMiner(ABC):
    """Abstract base class for PubMed mining"""

    def __init__(self, email: str, api_key: str, rate_limit: float = 0.34):
        """
        Initialize base miner

        Args:
            email: Email for NCBI API requests (required)
            rate_limit: Seconds between API requests (default: 0.34 for ~3 requests/second)
        """
        self.email = email
        self.rate_limit = rate_limit
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Common toxicity/side effect terms
        self.toxicity_terms = [
            "toxicity",
            "adverse effect",
            "side effect",
            "hepatotoxicity",
            "cardiotoxicity",
            "nephrotoxicity",
            "neurotoxicity",
            "cytotoxicity",
            "drug-induced",
            "adverse drug reaction",
            "ADR",
            "drug toxicity",
            "toxic effect",
            "adverse event",
            "safety profile",
            "contraindication",
            "drug-related",
            "medication-induced",
            "therapeutic toxicity",
        ]

        # Human protein patterns (improved regex)
        self.protein_patterns = [
            r"\b([A-Z][A-Z0-9]+)\b(?:\s+protein|\s+receptor|\s+enzyme|\s+kinase|\s+channel)",
            r"\b(CYP[0-9][A-Z][0-9]+)\b",  # Cytochrome P450 enzymes
            r"\b(P-gp|MDR1|ABCB1)\b",  # P-glycoprotein variants
            r"\b([A-Z]{2,}[0-9]+[A-Z]*)\b(?=\s+(?:protein|receptor|enzyme|kinase|channel|transporter))",
            r"\b(TNF-α|IL-[0-9]+|IFN-[αβγ])\b",  # Cytokines
            r"\b(EGFR|VEGFR|PDGFR|FGFR|IGF1R)\b",  # Growth factor receptors
            r"\b(BCL-2|BAX|p53|MDM2|BRCA[12])\b",  # Apoptosis/cancer related
            r"\b(GSTP1|GSTM1|GSTT1|NAT[12]|SULT[0-9][A-Z][0-9])\b",  # Metabolism enzymes
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.protein_patterns
        ]

    def _make_request(self, url: str, params: Dict) -> requests.Response:
        """Make rate-limited request to NCBI"""
        params["email"] = self.email
        params["api_key"] = self.api_key
        time.sleep(self.rate_limit)
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response

    def search_pubmed(self, query: str, max_results: int = 1000) -> List[str]:
        """Search PubMed and return PMIDs"""
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
        }

        response = self._make_request(search_url, params)
        root = ET.fromstring(response.content)

        pmids = []
        for pmid_elem in root.findall(".//Id"):
            pmids.append(pmid_elem.text)

        logger.info(f"Found {len(pmids)} PMIDs for query: {query}")
        return pmids

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """Fetch abstracts and metadata for given PMIDs"""
        if not pmids:
            return []

        # Process in batches to avoid URL length limits
        batch_size = 200
        all_papers = []

        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i : i + batch_size]
            pmid_str = ",".join(batch_pmids)

            fetch_url = f"{self.base_url}efetch.fcgi"
            params = {"db": "pubmed", "id": pmid_str, "retmode": "xml"}

            response = self._make_request(fetch_url, params)
            root = ET.fromstring(response.content)

            for article in root.findall(".//PubmedArticle"):
                paper_data = self._parse_article(article)
                if paper_data:
                    all_papers.append(paper_data)

        logger.info(f"Retrieved {len(all_papers)} abstracts")
        return all_papers

    def _parse_article(self, article) -> Optional[Dict]:
        """Parse XML article element into dictionary"""
        try:
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Extract abstract
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts)

            # Extract publication date
            pub_date = "Unknown"
            date_elem = article.find(".//PubDate")
            if date_elem is not None:
                year = date_elem.find("Year")
                month = date_elem.find("Month")
                if year is not None:
                    pub_date = year.text
                    if month is not None:
                        pub_date += f"-{month.text}"

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "publication_date": pub_date,
            }
        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None

    def extract_proteins_regex(self, text: str) -> List[str]:
        """Extract protein names using regex patterns"""
        proteins = set()

        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    proteins.update([m for m in match if m])
                else:
                    proteins.add(match)

        # Filter out common false positives
        false_positives = {
            "AND",
            "OR",
            "NOT",
            "THE",
            "FOR",
            "WITH",
            "ARE",
            "WAS",
            "WERE",
            "DNA",
            "RNA",
            "ATP",
            "ADP",
            "GDP",
            "GTP",
            "UDP",
            "UTP",
        }

        filtered_proteins = [
            p for p in proteins if p.upper() not in false_positives and len(p) > 2
        ]
        return sorted(list(set(filtered_proteins)))

    def extract_toxicity_terms(self, text: str) -> List[str]:
        """Extract toxicity-related terms from text"""
        found_terms = []
        text_lower = text.lower()

        for term in self.toxicity_terms:
            if term.lower() in text_lower:
                found_terms.append(term)

        return found_terms

    @abstractmethod
    def mine_proteins(
        self, query: str, max_results: int = 1000
    ) -> List[ProteinToxicityResult]:
        """Abstract method for mining proteins"""
        pass


class RegexPubMedMiner(BasePubMedMiner):
    """PubMed miner using regex-based protein extraction"""

    def __init__(self, email: str, api_key, rate_limit: float = 0.34):
        super().__init__(email, api_key, rate_limit)
        self.extraction_method = "regex"

    def mine_proteins(
        self, query: str = None, max_results: int = 1000
    ) -> List[ProteinToxicityResult]:
        """
        Mine proteins associated with drug toxicity using regex

        Args:
            query: Custom query (if None, uses default toxicity query)
            max_results: Maximum number of results to process
        """
        if query is None:
            # Default query focusing on drug toxicity and human proteins
            query = (
                "(drug toxicity OR adverse drug reaction OR side effect OR hepatotoxicity OR "
                "cardiotoxicity OR nephrotoxicity OR neurotoxicity) AND "
                "(protein OR enzyme OR receptor OR transporter) AND human"
            )

        # Search PubMed
        pmids = self.search_pubmed(query, max_results)

        # Fetch abstracts
        papers = self.fetch_abstracts(pmids)

        # Extract proteins and toxicity associations
        results = []
        for paper in papers:
            combined_text = f"{paper['title']} {paper['abstract']}"

            proteins = self.extract_proteins_regex(combined_text)
            toxicity_terms = self.extract_toxicity_terms(combined_text)

            # Only include papers with both proteins and toxicity terms
            if proteins and toxicity_terms:
                confidence_score = self._calculate_confidence_score(
                    proteins, toxicity_terms, combined_text
                )

                result = ProteinToxicityResult(
                    pmid=paper["pmid"],
                    title=paper["title"],
                    abstract=paper["abstract"],
                    proteins=proteins,
                    toxicity_terms=toxicity_terms,
                    confidence_score=confidence_score,
                    extraction_method=self.extraction_method,
                    publication_date=paper["publication_date"],
                )
                results.append(result)

        logger.info(f"Found {len(results)} papers with protein-toxicity associations")
        return results

    def _calculate_confidence_score(
        self, proteins: List[str], toxicity_terms: List[str], text: str
    ) -> float:
        """Calculate confidence score based on various factors"""
        score = 0.0

        # Base score for having proteins and toxicity terms
        score += 0.3

        # Bonus for multiple proteins
        score += min(len(proteins) * 0.1, 0.3)

        # Bonus for multiple toxicity terms
        score += min(len(toxicity_terms) * 0.1, 0.2)

        # Bonus for specific high-confidence terms
        high_confidence_terms = [
            "drug-induced",
            "adverse drug reaction",
            "hepatotoxicity",
            "cardiotoxicity",
        ]
        for term in high_confidence_terms:
            if term.lower() in text.lower():
                score += 0.1

        # Bonus for enzyme/metabolism terms
        if any(
            term in text.lower()
            for term in ["cyp", "metabolism", "enzyme", "metabolize"]
        ):
            score += 0.1

        return min(score, 1.0)


# class OnlineOllamaPubMedMiner(BasePubMedMiner):
#     """PubMed miner using Ollama LLM for protein extraction"""

#     def __init__(
#         self,
#         email: str,
#         ollama_url: str = "http://localhost:11434",
#         model: str = "llama3.2",
#         rate_limit: float = 0.34,
#     ):
#         super().__init__(email, rate_limit)
#         self.ollama_url = ollama_url
#         self.model = model
#         self.extraction_method = "ollama_llm"

#     def _query_ollama(self, prompt: str) -> str:
#         """Query Ollama LLM"""
#         try:
#             response = requests.post(
#                 f"{self.ollama_url}/api/generate",
#                 json={"model": self.model, "prompt": prompt, "stream": False},
#             )
#             response.raise_for_status()
#             return response.json().get("response", "")
#         except Exception as e:
#             logger.error(f"Error querying Ollama: {e}")
#             return ""

#     def extract_proteins_llm(self, text: str) -> Tuple[List[str], List[str], float]:
#         """Extract proteins using LLM with toxicity context"""
#         prompt = f"""
#         Analyze the following scientific text and extract:
#         1. Human protein names (including enzymes, receptors, transporters, channels)
#         2. Drug toxicity or side effect terms
#         3. Confidence score (0-1) for protein-toxicity associations

#         Focus on:
#         - Human proteins only (exclude non-human organisms)
#         - Standard protein nomenclature (e.g., CYP3A4, P-gp, TNF-α)
#         - Direct associations between proteins and toxicity

#         Text: {text}

#         Return your answer in this exact format:
#         PROTEINS: [protein1, protein2, ...]
#         TOXICITY_TERMS: [term1, term2, ...]
#         CONFIDENCE: [0.0-1.0]
#         """

#         response = self._query_ollama(prompt)

#         # Parse LLM response
#         proteins = []
#         toxicity_terms = []
#         confidence = 0.0

#         try:
#             lines = response.split("\n")
#             for line in lines:
#                 if line.startswith("PROTEINS:"):
#                     protein_text = line.replace("PROTEINS:", "").strip()
#                     proteins = self._parse_list_from_text(protein_text)
#                 elif line.startswith("TOXICITY_TERMS:"):
#                     toxicity_text = line.replace("TOXICITY_TERMS:", "").strip()
#                     toxicity_terms = self._parse_list_from_text(toxicity_text)
#                 elif line.startswith("CONFIDENCE:"):
#                     confidence_text = line.replace("CONFIDENCE:", "").strip()
#                     try:
#                         confidence = float(confidence_text)
#                     except ValueError:
#                         confidence = 0.5
#         except Exception as e:
#             logger.warning(f"Error parsing LLM response: {e}")
#             # Fallback to regex extraction
#             proteins = self.extract_proteins_regex(text)
#             toxicity_terms = self.extract_toxicity_terms(text)
#             confidence = 0.3

#         return proteins, toxicity_terms, confidence

#     def _parse_list_from_text(self, text: str) -> List[str]:
#         """Parse list from text (handles various formats)"""
#         # Remove brackets and split by comma
#         text = text.strip("[]")
#         items = [item.strip().strip("\"'") for item in text.split(",")]
#         return [item for item in items if item and item != "None"]

#     def mine_proteins(
#         self, query: str = None, max_results: int = 500
#     ) -> List[ProteinToxicityResult]:
#         """
#         Mine proteins using Ollama LLM analysis

#         Args:
#             query: Custom query (if None, uses default toxicity query)
#             max_results: Maximum number of results to process (lower default due to LLM processing time)
#         """
#         if query is None:
#             query = (
#                 "(drug toxicity OR adverse drug reaction OR side effect OR hepatotoxicity OR "
#                 "cardiotoxicity OR nephrotoxicity OR neurotoxicity) AND "
#                 "(protein OR enzyme OR receptor OR transporter) AND human"
#             )

#         # Search PubMed
#         pmids = self.search_pubmed(query, max_results)

#         # Fetch abstracts
#         papers = self.fetch_abstracts(pmids)

#         # Extract proteins using LLM
#         results = []
#         for i, paper in enumerate(papers):
#             logger.info(f"Processing paper {i+1}/{len(papers)} (PMID: {paper['pmid']})")

#             combined_text = f"{paper['title']} {paper['abstract']}"

#             proteins, toxicity_terms, confidence = self.extract_proteins_llm(
#                 combined_text
#             )

#             # Only include papers with both proteins and toxicity terms
#             if proteins and toxicity_terms:
#                 result = ProteinToxicityResult(
#                     pmid=paper["pmid"],
#                     title=paper["title"],
#                     abstract=paper["abstract"],
#                     proteins=proteins,
#                     toxicity_terms=toxicity_terms,
#                     confidence_score=confidence,
#                     extraction_method=self.extraction_method,
#                     publication_date=paper["publication_date"],
#                 )
#                 results.append(result)

#         logger.info(
#             f"Found {len(results)} papers with protein-toxicity associations using LLM"
#         )
#         return results


class LLMPubMedMiner(BasePubMedMiner):
    """PubMed miner using Hugging Face LLM for protein extraction"""

    def __init__(
        self,
        email: str,
        api_key: str,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",  # Or another local model
        rate_limit: float = 0.34,
        device: str = None,  # 'cuda', 'cpu', or 'mps'
    ):
        super().__init__(email, api_key, rate_limit)
        self.model_name = model_name
        self.extraction_method = "huggingface_llm"

        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 #if self.device == "cuda" else torch.float32,
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # device=0 if self.device == "cuda" else -1,
        )

    def _query_llm(self, prompt: str) -> str:
        """Generate text using a local Hugging Face LLM"""
        if self.model_name.contains("mistral"):
            prompt = f"[INST]{prompt}[/INST]"
        try:
            outputs = self.generator(
                prompt, max_new_tokens=512, do_sample=True, temperature=0.7
            )
            logger.info(f"Raw LLM output: {outputs}")
            return outputs[0]["generated_text"].replace(prompt, "").strip()
        except Exception as e:
            logger.error(f"Error running Hugging Face model: {e}")
            return ""

    def extract_proteins_llm(self, text: str) -> Tuple[List[str], List[str], float]:
        """Extract proteins using LLM with toxicity context"""
        prompt = f"""
        Analyze the following scientific text and extract:
        1. Human protein names (including enzymes, receptors, transporters, channels)
        2. Drug toxicity or side effect terms
        3. Confidence score (0-1) for protein-toxicity associations

        Focus on:
        - Human proteins only (exclude non-human organisms)
        - Standard protein nomenclature (e.g., CYP3A4, P-gp, TNF-α)
        - Direct associations between proteins and toxicity

        Text: {text}

        Return your answer in this exact format:
        PROTEINS: [protein1, protein2, ...]
        TOXICITY_TERMS: [term1, term2, ...]
        CONFIDENCE: [0.0-1.0]
        """

        response = self._query_llm(prompt)

        # Parse LLM response
        proteins = []
        toxicity_terms = []
        confidence = 0.0

        try:
            lines = response.split("\n")
            for line in lines:
                if line.startswith("PROTEINS:"):
                    protein_text = line.replace("PROTEINS:", "").strip()
                    proteins = self._parse_list_from_text(protein_text)
                elif line.startswith("TOXICITY_TERMS:"):
                    toxicity_text = line.replace("TOXICITY_TERMS:", "").strip()
                    toxicity_terms = self._parse_list_from_text(toxicity_text)
                elif line.startswith("CONFIDENCE:"):
                    confidence_text = line.replace("CONFIDENCE:", "").strip()
                    try:
                        confidence = float(confidence_text)
                    except ValueError:
                        confidence = 0.5
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            # Fallback to regex extraction
            proteins = self.extract_proteins_regex(text)
            toxicity_terms = self.extract_toxicity_terms(text)
            confidence = 0.3

        return proteins, toxicity_terms, confidence

    def _parse_list_from_text(self, text: str) -> List[str]:
        """Parse list from text (handles various formats)"""
        # Remove brackets and split by comma
        text = text.strip("[]")
        items = [item.strip().strip("\"'") for item in text.split(",")]
        return [item for item in items if item and item != "None"]

    def mine_proteins(
        self, query: str = None, max_results: int = 500
    ) -> List[ProteinToxicityResult]:
        """
        Mine proteins using LLM analysis

        Args:
            query: Custom query (if None, uses default toxicity query)
            max_results: Maximum number of results to process (lower default due to LLM processing time)
        """
        if query is None:
            query = (
                "(drug toxicity OR adverse drug reaction OR side effect OR hepatotoxicity OR "
                "cardiotoxicity OR nephrotoxicity OR neurotoxicity) AND "
                "(protein OR enzyme OR receptor OR transporter) AND human"
            )

        # Search PubMed
        pmids = self.search_pubmed(query, max_results)

        # Fetch abstracts
        papers = self.fetch_abstracts(pmids)

        # Extract proteins using LLM
        results = []
        for i, paper in enumerate(papers):
            logger.info(f"Processing paper {i+1}/{len(papers)} (PMID: {paper['pmid']})")

            combined_text = f"{paper['title']} {paper['abstract']}"

            proteins, toxicity_terms, confidence = self.extract_proteins_llm(
                combined_text
            )

            # Only include papers with both proteins and toxicity terms
            if proteins and toxicity_terms:
                result = ProteinToxicityResult(
                    pmid=paper["pmid"],
                    title=paper["title"],
                    abstract=paper["abstract"],
                    proteins=proteins,
                    toxicity_terms=toxicity_terms,
                    confidence_score=confidence,
                    extraction_method=self.extraction_method,
                    publication_date=paper["publication_date"],
                )
                results.append(result)

        logger.info(
            f"Found {len(results)} papers with protein-toxicity associations using LLM"
        )
        return results

    # def mine_proteins(self, text: str) -> Tuple[List[str], List[str], float]:
    #     """Extract proteins using LLM with toxicity context"""
    #     prompt = f"""
    #     Analyze the following scientific text and extract:
    #     1. Human protein names (including enzymes, receptors, transporters, channels)
    #     2. Drug toxicity or side effect terms
    #     3. Confidence score (0-1) for protein-toxicity associations
        
    #     Focus on:
    #     - Human proteins only (exclude non-human organisms)
    #     - Standard protein nomenclature (e.g., CYP3A4, P-gp, TNF-α)
    #     - Direct associations between proteins and toxicity
        
    #     Text: {text}
        
    #     Return your answer in this exact format:
    #     PROTEINS: [protein1, protein2, ...]
    #     TOXICITY_TERMS: [term1, term2, ...]
    #     CONFIDENCE: [0.0-1.0]
    #     """

    #     response = self._query_llm(prompt)

    #     proteins = []
    #     toxicity_terms = []
    #     confidence = 0.0

    #     try:
    #         lines = response.split("\n")
    #         for line in lines:
    #             if line.startswith("PROTEINS:"):
    #                 protein_text = line.replace("PROTEINS:", "").strip()
    #                 proteins = self._parse_list_from_text(protein_text)
    #             elif line.startswith("TOXICITY_TERMS:"):
    #                 toxicity_text = line.replace("TOXICITY_TERMS:", "").strip()
    #                 toxicity_terms = self._parse_list_from_text(toxicity_text)
    #             elif line.startswith("CONFIDENCE:"):
    #                 confidence_text = line.replace("CONFIDENCE:", "").strip()
    #                 confidence = float(confidence_text)
    #     except Exception as e:
    #         logger.warning(f"Error parsing LLM response: {e}")
    #         proteins = self.extract_proteins_regex(text)
    #         toxicity_terms = self.extract_toxicity_terms(text)
    #         confidence = 0.3

    #     return proteins, toxicity_terms, confidence


class ProteinToxicityAnalyzer:
    """Analyzer for protein-toxicity mining results"""

    def __init__(self):
        pass

    def results_to_dataframe(
        self, results: List[ProteinToxicityResult]
    ) -> pl.DataFrame:
        """Convert results to Polars DataFrame"""
        data = []
        for result in results:
            for protein in result.proteins:
                for toxicity_term in result.toxicity_terms:
                    data.append(
                        {
                            "pmid": result.pmid,
                            "title": result.title,
                            "protein": protein,
                            "toxicity_term": toxicity_term,
                            "confidence_score": result.confidence_score,
                            "extraction_method": result.extraction_method,
                            "publication_date": result.publication_date,
                            "abstract": result.abstract,
                        }
                    )

        return pl.DataFrame(data)

    def analyze_protein_toxicity_associations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Analyze protein-toxicity associations"""
        # Group by protein and toxicity term
        associations = (
            df.group_by(["protein", "toxicity_term"])
            .agg(
                [
                    pl.len().alias("paper_count"),
                    pl.mean("confidence_score").alias("avg_confidence"),
                    pl.col("pmid").unique().len().alias("unique_papers"),
                    pl.col("extraction_method").first().alias("extraction_method"),
                ]
            )
            .sort("paper_count", descending=True)
        )

        return associations

    def get_top_toxic_proteins(self, df: pl.DataFrame, top_n: int = 20) -> pl.DataFrame:
        """Get proteins most associated with toxicity"""
        protein_stats = (
            df.group_by("protein")
            .agg(
                [
                    pl.len().alias("total_associations"),
                    pl.mean("confidence_score").alias("avg_confidence"),
                    pl.col("pmid").unique().len().alias("unique_papers"),
                    pl.col("toxicity_term")
                    .unique()
                    .len()
                    .alias("unique_toxicity_terms"),
                ]
            )
            .sort("total_associations", descending=True)
            .head(top_n)
        )

        return protein_stats

    def save_results(self, results: List[ProteinToxicityResult], filename: str):
        """Save results to JSON file"""
        serializable_results = []
        for result in results:
            serializable_results.append(
                {
                    "pmid": result.pmid,
                    "title": result.title,
                    "abstract": result.abstract,
                    "proteins": result.proteins,
                    "toxicity_terms": result.toxicity_terms,
                    "confidence_score": result.confidence_score,
                    "extraction_method": result.extraction_method,
                    "publication_date": result.publication_date,
                }
            )

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {filename}")


def main():

    email = sys.argv[1]
    api_key = sys.argv[2]
    rate = 0.1


    llm_miner = LLMPubMedMiner(email=email, api_key=api_key,rate_limit=.1)
    print(llm_miner._query_llm("An adverse drug reaction is defined as"))
    print("should be done")
    exit(0)
    # # Example 1: Regex-based mining
    # regex_miner = RegexPubMedMiner(email=email, api_key=api_key, rate_limit=0.1)
    # regex_results = regex_miner.mine_proteins(max_results=100000)

    # Example 2: LLM-based mining (requires LLM running locally)
    try:
        llm_miner = LLMPubMedMiner(
            email=email, api_key=api_key, rate_limit=0.1,
        )
        llm_results = llm_miner.mine_proteins(max_results=100000)
    except Exception as e:
        logger.error(f"LLM mining failed: {e}")
        llm_results = []

    # Analyze results
    analyzer = ProteinToxicityAnalyzer()

    # # Convert to DataFrame and analyze
    # if regex_results:
    #     regex_df = analyzer.results_to_dataframe(regex_results)
    #     top_proteins = analyzer.get_top_toxic_proteins(regex_df)
    #     print("Top toxic proteins (Regex method):")
    #     print(top_proteins)

    #     # Save results
    #     analyzer.save_results(regex_results, "regex_protein_toxicity_results.json")

    if llm_results:
        llm_df = analyzer.results_to_dataframe(llm_results)
        print("\nTop toxic proteins (LLM method):")
        print(analyzer.get_top_toxic_proteins(llm_df))

        # Save results
        analyzer.save_results(llm_results, "llm_protein_toxicity_results.json")


# Example usage
if __name__ == "__main__":
    main()


#     def process_literature_corpus(self, corpus_path: str, corpus_type: str = "auto") -> List[Dict]:
#         """
#         Process literature corpus from various sources.

#         Args:
#             corpus_path: Path to corpus (file or directory)
#             corpus_type: Type of corpus ('txt', 'json', 'csv', 'pdf', 'auto')

#         Returns:
#             List of processed documents
#         """
#         print(f"Processing literature corpus from: {corpus_path}")

#         documents = []

#         if corpus_type == "auto":
#             # Auto-detect corpus type
#             if os.path.isdir(corpus_path):
#                 # Check for different file types in directory
#                 txt_files = glob.glob(os.path.join(corpus_path, "*.txt"))
#                 pdf_files = glob.glob(os.path.join(corpus_path, "*.pdf"))
#                 json_files = glob.glob(os.path.join(corpus_path, "*.json"))

#                 if txt_files:
#                     documents.extend(self.corpus_processor.read_text_files(corpus_path, "*.txt"))
#                 if pdf_files:
#                     documents.extend(self.corpus_processor.process_pdf_directory(corpus_path))
#                 if json_files:
#                     for json_file in json_files:
#                         documents.extend(self.corpus_processor.read_json_corpus(json_file))

#             elif os.path.isfile(corpus_path):
#                 _, ext = os.path.splitext(corpus_path)
#                 if ext.lower() == '.json':
#                     documents = self.corpus_processor.read_json_corpus(corpus_path)
#                 elif ext.lower() == '.csv':
#                     documents = self.corpus_processor.read_csv_corpus(corpus_path)
#                 elif ext.lower() == '.txt':
#                     documents = self.corpus_processor.read_text_files(os.path.dirname(corpus_path),
#                                                                      os.path.basename(corpus_path))

#         elif corpus_type == "txt":
#             if os.path.isdir(corpus_path):
#                 documents = self.corpus_processor.read_text_files(corpus_path)
#             else:
#                 documents = self.corpus_processor.read_text_files(os.path.dirname(corpus_path),
#                                                                  os.path.basename(corpus_path))

#         elif corpus_type == "json":
#             documents = self.corpus_processor.read_json_corpus(corpus_path)

#         elif corpus_type == "csv":
#             documents = self.corpus_processor.read_csv_corpus(corpus_path)

#         elif corpus_type == "pdf":
#             if os.path.isdir(corpus_path):
#                 documents = self.corpus_processor.process_pdf_directory(corpus_path)

#         print(f"Loaded {len(documents)} documents from corpus")
#         return documents

#     def extract_proteins_and_side_effects_dual(self, documents: List[Dict]) -> None:
#         """
#         Extract proteins and side effects using both regex and LLM approaches.

#         Args:
#             documents: List of document dictionaries
#         """
#         print("Extracting proteins and side effects using dual approach...")

#         for i, doc in enumerate(documents):
#             doc_id = doc.get('id', f"doc_{i}")
#             text = doc.get('content', '')
#             title = doc.get('title', '')
#             full_text = f"{title} {text}"

#             if not text.strip():
#                 continue

#             print(f"Processing document {i+1}/{len(documents)}: {doc_id}")

#             # === REGEX-based extraction ===
#             regex_proteins = set()
#             regex_side_effects = []

#             # Extract side effects using keywords
#             text_lower = full_text.lower()
#             for keyword in self.side_effect_keywords:
#                 if keyword in text_lower:
#                     regex_side_effects.append(keyword)

#             # Extract proteins using regex patterns
#             for pattern in self.protein_patterns:
#                 matches = re.findall(pattern, full_text)
#                 regex_proteins.update(matches)

#             # Use spaCy for additional protein extraction
#             if self.nlp:
#                 doc_nlp = self.nlp(full_text)
#                 for ent in doc_nlp.ents:
#                     if ent.label_ in ["PERSON", "ORG"] and len(ent.text) < 15:
#                         if re.match(r'^[A-Z][a-z]*[0-9]*[A-Z]*#!/usr/bin/env python3
# """
# PubMed Protein Side Effects Text Miner
# ======================================

# This tool mines PubMed abstracts for proteins related to side effects,
# maps them to UniProt identifiers, and extracts protein networks using StringDB.

# Dependencies:
# pip install biopython requests pandas networkx matplotlib seaborn nltk spacy

# You'll also need to download the spaCy model:
# python -m spacy download en_core_web_sm

# Author: Claude
# """

# import requests
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns
# from Bio import Entrez
# import re
# import json
# import time
# import os
# import glob
# from pathlib import Path
# from collections import defaultdict, Counter
# from typing import List, Dict, Set, Tuple, Optional, Union
# import warnings
# warnings.filterwarnings('ignore')

# # NLP imports
# import nltk
# import spacy
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')

# class OllamaLLMExtractor:
#     """
#     Local LLM-based protein and side effect extractor using Ollama.
#     """

#     def __init__(self, model_name: str = "llama3.1:8b", ollama_host: str = "http://localhost:11434"):
#         """
#         Initialize the Ollama LLM extractor.

#         Args:
#             model_name: Name of the Ollama model to use
#             ollama_host: Ollama server host URL
#         """
#         self.model_name = model_name
#         self.ollama_host = ollama_host
#         self.available_models = self._get_available_models()

#         # Extraction prompts
#         self.protein_extraction_prompt = """
# You are a biomedical text mining expert. Extract all protein names, gene names, and enzyme names from the following text.

# Important instructions:
# 1. Include common protein names (e.g., p53, TNF-α, VEGF)
# 2. Include gene names (e.g., BRCA1, EGFR, MYC)
# 3. Include enzyme names (e.g., cytochrome P450, kinase names)
# 4. Include protein complexes (e.g., NF-κB, AP-1)
# 5. Only extract names that clearly refer to proteins/genes/enzymes
# 6. Do not extract drug names, disease names, or general biological terms
# 7. Return results as a JSON list of unique protein names

# Text to analyze:
# {text}

# Return only the JSON list, no other text:
# """

#         self.side_effect_extraction_prompt = """
# You are a medical text mining expert. Extract all adverse effects, side effects, toxicities, and harmful reactions mentioned in the following text.

# Important instructions:
# 1. Include specific adverse effects (e.g., hepatotoxicity, nephrotoxicity)
# 2. Include general side effects (e.g., nausea, headache, rash)
# 3. Include toxic effects (e.g., cardiotoxicity, neurotoxicity)
# 4. Include allergic reactions and hypersensitivity
# 5. Include organ damage or dysfunction
# 6. Normalize similar terms (e.g., "liver toxicity" → "hepatotoxicity")
# 7. Return results as a JSON list of unique side effect terms

# Text to analyze:
# {text}

# Return only the JSON list, no other text:
# """

#         self.protein_side_effect_link_prompt = """
# You are a biomedical expert. Analyze the following text and identify relationships between proteins and side effects/adverse events.

# Extract information in this format:
# - Protein name
# - Associated side effect
# - Relationship type (causes, associated_with, biomarker_for, protective_against)
# - Confidence (high, medium, low)

# Text to analyze:
# {text}

# Return results as JSON in this format:
# [
#   {{
#     "protein": "protein_name",
#     "side_effect": "side_effect_name",
#     "relationship": "relationship_type",
#     "confidence": "confidence_level"
#   }}
# ]

# Return only the JSON, no other text:
# """

#     def _get_available_models(self) -> List[str]:
#         """Get list of available Ollama models."""
#         try:
#             response = requests.get(f"{self.ollama_host}/api/tags")
#             if response.status_code == 200:
#                 models = response.json()
#                 return [model['name'] for model in models['models']]
#             return []
#         except Exception as e:
#             print(f"Warning: Could not connect to Ollama server: {e}")
#             return []

#     def _query_ollama(self, prompt: str, max_retries: int = 3) -> str:
#         """
#         Query Ollama API with retry logic.

#         Args:
#             prompt: The prompt to send to the model
#             max_retries: Maximum number of retries

#         Returns:
#             Model response text
#         """
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     f"{self.ollama_host}/api/generate",
#                     json={
#                         "model": self.model_name,
#                         "prompt": prompt,
#                         "stream": False,
#                         "options": {
#                             "temperature": 0.1,  # Low temperature for consistent extraction
#                             "top_p": 0.9,
#                             "num_predict": 2000
#                         }
#                     },
#                     timeout=120
#                 )

#                 if response.status_code == 200:
#                     return response.json()['response']
#                 else:
#                     print(f"Ollama API error: {response.status_code}")

#             except Exception as e:
#                 print(f"Ollama query attempt {attempt + 1} failed: {e}")
#                 if attempt < max_retries - 1:
#                     time.sleep(2 ** attempt)  # Exponential backoff

#         return ""

#     def extract_proteins_llm(self, text: str) -> List[str]:
#         """
#         Extract proteins using LLM.

#         Args:
#             text: Input text to analyze

#         Returns:
#             List of extracted protein names
#         """
#         prompt = self.protein_extraction_prompt.format(text=text)
#         response = self._query_ollama(prompt)

#         try:
#             # Try to parse JSON response
#             proteins = json.loads(response.strip())
#             if isinstance(proteins, list):
#                 return [p.strip() for p in proteins if p.strip()]
#             return []
#         except json.JSONDecodeError:
#             # Fallback: extract from text response
#             proteins = []
#             lines = response.split('\n')
#             for line in lines:
#                 line = line.strip()
#                 if line.startswith('-') or line.startswith('•'):
#                     protein = line[1:].strip()
#                     if protein:
#                         proteins.append(protein)
#             return proteins

#     def extract_side_effects_llm(self, text: str) -> List[str]:
#         """
#         Extract side effects using LLM.

#         Args:
#             text: Input text to analyze

#         Returns:
#             List of extracted side effects
#         """
#         prompt = self.side_effect_extraction_prompt.format(text=text)
#         response = self._query_ollama(prompt)

#         try:
#             side_effects = json.loads(response.strip())
#             if isinstance(side_effects, list):
#                 return [s.strip() for s in side_effects if s.strip()]
#             return []
#         except json.JSONDecodeError:
#             side_effects = []
#             lines = response.split('\n')
#             for line in lines:
#                 line = line.strip()
#                 if line.startswith('-') or line.startswith('•'):
#                     effect = line[1:].strip()
#                     if effect:
#                         side_effects.append(effect)
#             return side_effects

#     def extract_protein_side_effect_links_llm(self, text: str) -> List[Dict]:
#         """
#         Extract protein-side effect relationships using LLM.

#         Args:
#             text: Input text to analyze

#         Returns:
#             List of relationship dictionaries
#         """
#         prompt = self.protein_side_effect_link_prompt.format(text=text)
#         response = self._query_ollama(prompt)

#         try:
#             relationships = json.loads(response.strip())
#             if isinstance(relationships, list):
#                 return relationships
#             return []
#         except json.JSONDecodeError:
#             return []

# class LiteratureCorpusProcessor:
#     """
#     Process various literature formats (PDF, TXT, JSON, etc.) for text mining.
#     """

#     def __init__(self):
#         """Initialize the corpus processor."""
#         self.supported_formats = ['.txt', '.json', '.csv', '.pdf']

#     def read_text_files(self, directory: str, pattern: str = "*.txt") -> List[Dict]:
#         """
#         Read text files from a directory.

#         Args:
#             directory: Directory containing text files
#             pattern: File pattern to match

#         Returns:
#             List of document dictionaries
#         """
#         documents = []
#         files = glob.glob(os.path.join(directory, pattern))

#         for file_path in files:
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     documents.append({
#                         'id': os.path.basename(file_path),
#                         'source': file_path,
#                         'content': content,
#                         'title': os.path.basename(file_path)
#                     })
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")

#         return documents

#     def read_json_corpus(self, file_path: str) -> List[Dict]:
#         """
#         Read JSON corpus file.

#         Args:
#             file_path: Path to JSON file

#         Returns:
#             List of document dictionaries
#         """
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             if isinstance(data, list):
#                 return data
#             elif isinstance(data, dict):
#                 return [data]
#             return []
#         except Exception as e:
#             print(f"Error reading JSON corpus: {e}")
#             return []

#     def read_csv_corpus(self, file_path: str, text_column: str = 'text',
#                        id_column: str = 'id', title_column: str = 'title') -> List[Dict]:
#         """
#         Read CSV corpus file.

#         Args:
#             file_path: Path to CSV file
#             text_column: Column name containing text
#             id_column: Column name containing document ID
#             title_column: Column name containing title

#         Returns:
#             List of document dictionaries
#         """
#         try:
#             df = pd.read_csv(file_path)
#             documents = []

#             for _, row in df.iterrows():
#                 doc = {
#                     'id': row.get(id_column, f"doc_{len(documents)}"),
#                     'source': file_path,
#                     'content': row.get(text_column, ''),
#                     'title': row.get(title_column, row.get(id_column, ''))
#                 }
#                 documents.append(doc)

#             return documents
#         except Exception as e:
#             print(f"Error reading CSV corpus: {e}")
#             return []

#     def process_pdf_directory(self, directory: str) -> List[Dict]:
#         """
#         Process PDF files in a directory (requires PyPDF2).

#         Args:
#             directory: Directory containing PDF files

#         Returns:
#             List of document dictionaries
#         """
#         documents = []

#         try:
#             import PyPDF2

#             pdf_files = glob.glob(os.path.join(directory, "*.pdf"))

#             for pdf_path in pdf_files:
#                 try:
#                     with open(pdf_path, 'rb') as file:
#                         pdf_reader = PyPDF2.PdfReader(file)
#                         text = ""

#                         for page in pdf_reader.pages:
#                             text += page.extract_text()

#                         documents.append({
#                             'id': os.path.basename(pdf_path),
#                             'source': pdf_path,
#                             'content': text,
#                             'title': os.path.basename(pdf_path)
#                         })

#                 except Exception as e:
#                     print(f"Error processing PDF {pdf_path}: {e}")

#         except ImportError:
#             print("PyPDF2 not installed. Install with: pip install PyPDF2")

#         return documents
#     """
#     A comprehensive text mining tool for extracting proteins related to side effects
#     from PubMed abstracts with UniProt mapping and StringDB network analysis.
#     """

#     def __init__(self, email: str, max_retries: int = 3):
#         """
#         Initialize the miner with configuration.

#         Args:
#             email: Email for NCBI API access (required)
#             max_retries: Maximum number of retries for API calls
#         """
#         self.email = email
#         self.max_retries = max_retries
#         Entrez.email = email

#         # Load spaCy model for NER
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
#             self.nlp = None

#         # Initialize data storage
#         self.abstracts = []
#         self.protein_mentions = defaultdict(list)
#         self.side_effect_mentions = defaultdict(list)
#         self.protein_uniprot_mapping = {}
#         self.string_networks = {}

#         # Define side effect keywords
#         self.side_effect_keywords = {
#             'adverse effects', 'side effects', 'toxicity', 'adverse reactions',
#             'adverse events', 'hepatotoxicity', 'nephrotoxicity', 'cardiotoxicity',
#             'neurotoxicity', 'cytotoxicity', 'genotoxicity', 'mutagenicity',
#             'carcinogenicity', 'teratogenicity', 'immunotoxicity', 'allergic reaction',
#             'hypersensitivity', 'anaphylaxis', 'rash', 'nausea', 'vomiting',
#             'diarrhea', 'headache', 'dizziness', 'fatigue', 'insomnia',
#             'depression', 'anxiety', 'seizure', 'tremor', 'muscle pain',
#             'joint pain', 'fever', 'chills', 'weight gain', 'weight loss',
#             'hair loss', 'skin reactions', 'liver damage', 'kidney damage',
#             'heart problems', 'blood disorders', 'bone marrow suppression'
#         }

#         # Common protein name patterns
#         self.protein_patterns = [
#             r'\b[A-Z][a-z]+\d+[a-z]*\b',  # e.g., Cyp3a4, Tnf1a
#             r'\b[A-Z]{2,}[0-9]*[A-Z]*\b',  # e.g., TNF, IL1B, VEGFA
#             r'\b[A-Z][a-z]+-?[A-Z0-9]+\b',  # e.g., p53, NF-kB
#             r'\b[A-Z]{1,2}[0-9]+[A-Z]*\b',  # e.g., P53, IL6
#         ]

#     def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
#         """
#         Search PubMed for articles matching the query.

#         Args:
#             query: Search query string
#             max_results: Maximum number of results to retrieve

#         Returns:
#             List of PubMed IDs
#         """
#         print(f"Searching PubMed for: {query}")

#         try:
#             # Search for articles
#             search_handle = Entrez.esearch(
#                 db="pubmed",
#                 term=query,
#                 retmax=max_results,
#                 sort="pub_date"
#             )
#             search_results = Entrez.read(search_handle)
#             search_handle.close()

#             pmids = search_results["IdList"]
#             print(f"Found {len(pmids)} articles")
#             return pmids

#         except Exception as e:
#             print(f"Error searching PubMed: {e}")
#             return []

#     def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
#         """
#         Fetch abstracts for given PubMed IDs.

#         Args:
#             pmids: List of PubMed IDs

#         Returns:
#             List of dictionaries containing abstract information
#         """
#         print(f"Fetching abstracts for {len(pmids)} articles...")
#         abstracts = []

#         # Process in batches to avoid overwhelming the API
#         batch_size = 20
#         for i in range(0, len(pmids), batch_size):
#             batch = pmids[i:i + batch_size]

#             try:
#                 # Fetch abstracts
#                 fetch_handle = Entrez.efetch(
#                     db="pubmed",
#                     id=",".join(batch),
#                     rettype="abstract",
#                     retmode="xml"
#                 )
#                 fetch_results = Entrez.read(fetch_handle)
#                 fetch_handle.close()

#                 # Parse results
#                 for article in fetch_results["PubmedArticle"]:
#                     try:
#                         medline_citation = article["MedlineCitation"]
#                         pmid = str(medline_citation["PMID"])

#                         # Extract title
#                         title = medline_citation["Article"]["ArticleTitle"]

#                         # Extract abstract
#                         abstract_text = ""
#                         if "Abstract" in medline_citation["Article"]:
#                             abstract_sections = medline_citation["Article"]["Abstract"]["AbstractText"]
#                             if isinstance(abstract_sections, list):
#                                 abstract_text = " ".join([str(section) for section in abstract_sections])
#                             else:
#                                 abstract_text = str(abstract_sections)

#                         # Extract publication date
#                         pub_date = "Unknown"
#                         if "PubDate" in medline_citation["Article"]["Journal"]["JournalIssue"]:
#                             date_info = medline_citation["Article"]["Journal"]["JournalIssue"]["PubDate"]
#                             if "Year" in date_info:
#                                 pub_date = str(date_info["Year"])

#                         abstracts.append({
#                             "pmid": pmid,
#                             "title": title,
#                             "abstract": abstract_text,
#                             "pub_date": pub_date
#                         })

#                     except Exception as e:
#                         print(f"Error processing article: {e}")
#                         continue

#                 # Be respectful to NCBI servers
#                 time.sleep(0.5)

#             except Exception as e:
#                 print(f"Error fetching batch: {e}")
#                 continue

#         print(f"Successfully fetched {len(abstracts)} abstracts")
#         return abstracts

#     def extract_proteins_and_side_effects(self, abstracts: List[Dict]) -> None:
#         """
#         Extract protein mentions and side effects from abstracts (legacy method for PubMed).

#         Args:
#             abstracts: List of abstract dictionaries
#         """
#         print("Extracting proteins and side effects from PubMed abstracts...")

#         # Convert PubMed abstracts to document format
#         documents = []
#         for abstract in abstracts:
#             documents.append({
#                 'id': abstract['pmid'],
#                 'content': abstract['abstract'],
#                 'title': abstract['title']
#             })

#         # Use the dual extraction method
#         self.extract_proteins_and_side_effects_dual(documents)

#     def map_to_uniprot(self, proteins: Set[str]) -> Dict[str, str]:
#         """
#         Map protein names to UniProt identifiers.

#         Args:
#             proteins: Set of protein names

#         Returns:
#             Dictionary mapping protein names to UniProt IDs
#         """
#         print(f"Mapping {len(proteins)} proteins to UniProt...")
#         mapping = {}

#         for protein in proteins:
#             try:
#                 # Query UniProt API
#                 url = f"https://rest.uniprot.org/uniprotkb/search"
#                 params = {
#                     'query': f'gene:{protein} OR protein_name:{protein}',
#                     'format': 'json',
#                     'size': 5,
#                     'fields': 'accession,gene_names,protein_name'
#                 }

#                 response = requests.get(url, params=params, timeout=10)

#                 if response.status_code == 200:
#                     data = response.json()
#                     if data['results']:
#                         # Take the first result
#                         result = data['results'][0]
#                         accession = result['primaryAccession']
#                         mapping[protein] = accession

#                         # Store additional info
#                         self.protein_uniprot_mapping[protein] = {
#                             'accession': accession,
#                             'gene_names': result.get('genes', []),
#                             'protein_name': result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')
#                         }

#                 # Be respectful to the API
#                 time.sleep(0.1)

#             except Exception as e:
#                 print(f"Error mapping {protein}: {e}")
#                 continue

#         print(f"Successfully mapped {len(mapping)} proteins to UniProt")
#         return mapping

#     def get_string_network(self, uniprot_ids: List[str], species: str = "9606") -> Dict:
#         """
#         Retrieve protein-protein interaction network from StringDB.

#         Args:
#             uniprot_ids: List of UniProt identifiers
#             species: Species taxonomy ID (default: 9606 for human)

#         Returns:
#             Dictionary containing network information
#         """
#         print(f"Retrieving StringDB network for {len(uniprot_ids)} proteins...")

#         try:
#             # StringDB API endpoint
#             string_api_url = "https://string-db.org/api"

#             # Get string identifiers
#             url = f"{string_api_url}/json/get_string_ids"
#             params = {
#                 'identifiers': '\r'.join(uniprot_ids),
#                 'species': species,
#                 'limit': 1,
#                 'echo_query': 1
#             }

#             response = requests.post(url, data=params)
#             string_ids = response.json()

#             # Map UniProt to String IDs
#             uniprot_to_string = {}
#             for item in string_ids:
#                 if 'stringId' in item:
#                     uniprot_to_string[item['queryItem']] = item['stringId']

#             if not uniprot_to_string:
#                 print("No String IDs found for the provided proteins")
#                 return {}

#             # Get network interactions (excluding text-mining evidence)
#             url = f"{string_api_url}/json/network"
#             params = {
#                 'identifiers': '\r'.join(uniprot_to_string.values()),
#                 'species': species,
#                 'required_score': 400,  # Medium confidence
#                 'hide_disconnected_nodes': 1,
#                 'network_flavor': 'evidence'
#             }

#             response = requests.post(url, data=params)
#             network_data = response.json()

#             # Filter out text-mining evidence
#             filtered_interactions = []
#             for interaction in network_data:
#                 evidence_types = []
#                 for evidence in ['neighborhood', 'fusion', 'cooccurence', 'coexpression',
#                                'experimental', 'database', 'textmining']:
#                     if interaction.get(evidence, 0) > 0:
#                         evidence_types.append(evidence)

#                 # Exclude interactions based only on text-mining
#                 if evidence_types and 'textmining' in evidence_types:
#                     non_textmining = [e for e in evidence_types if e != 'textmining']
#                     if non_textmining:  # Has non-text-mining evidence
#                         interaction['evidence_types'] = non_textmining
#                         filtered_interactions.append(interaction)
#                 elif evidence_types:  # Has evidence and no text-mining
#                     interaction['evidence_types'] = evidence_types
#                     filtered_interactions.append(interaction)

#             print(f"Found {len(filtered_interactions)} protein-protein interactions")

#             return {
#                 'uniprot_to_string': uniprot_to_string,
#                 'interactions': filtered_interactions
#             }

#         except Exception as e:
#             print(f"Error retrieving StringDB network: {e}")
#             return {}

#     def create_network_graph(self, network_data: Dict) -> nx.Graph:
#         """
#         Create a NetworkX graph from StringDB network data.

#         Args:
#             network_data: Network data from StringDB

#         Returns:
#             NetworkX graph object
#         """
#         G = nx.Graph()

#         if not network_data or 'interactions' not in network_data:
#             return G

#         # Add nodes and edges
#         for interaction in network_data['interactions']:
#             node1 = interaction['preferredName_A']
#             node2 = interaction['preferredName_B']
#             score = interaction['score']
#             evidence = interaction.get('evidence_types', [])

#             # Add nodes
#             G.add_node(node1)
#             G.add_node(node2)

#             # Add edge with attributes
#             G.add_edge(node1, node2, weight=score, evidence=evidence)

#         return G

#     def visualize_network(self, G: nx.Graph, title: str = "Protein-Protein Interaction Network"):
#         """
#         Visualize the protein interaction network.

#         Args:
#             G: NetworkX graph
#             title: Plot title
#         """
#         if len(G.nodes()) == 0:
#             print("No network data to visualize")
#             return

#         plt.figure(figsize=(12, 8))

#         # Layout
#         pos = nx.spring_layout(G, k=3, iterations=50)

#         # Draw network
#         nx.draw_networkx_nodes(G, pos, node_color='lightblue',
#                               node_size=1000, alpha=0.7)
#         nx.draw_networkx_labels(G, pos, font_size=8)

#         # Draw edges with varying thickness based on score
#         edges = G.edges()
#         weights = [G[u][v]['weight'] for u, v in edges]
#         nx.draw_networkx_edges(G, pos, width=[w/200 for w in weights],
#                               alpha=0.5, edge_color='gray')

#         plt.title(title)
#         plt.axis('off')
#         plt.tight_layout()
#         plt.show()

#     def analyze_results(self) -> pd.DataFrame:
#         """
#         Analyze and summarize the mining results with method comparison.

#         Returns:
#             DataFrame with analysis results
#         """
#         print("Analyzing results...")

#         # Count protein mentions
#         all_proteins = []
#         for doc_id, proteins in self.protein_mentions.items():
#             all_proteins.extend(proteins)

#         protein_counts = Counter(all_proteins)

#         # Count side effects
#         all_side_effects = []
#         for doc_id, effects in self.side_effect_mentions.items():
#             all_side_effects.extend(effects)

#         side_effect_counts = Counter(all_side_effects)

#         # Create analysis DataFrame
#         analysis_data = []
#         for protein, count in protein_counts.most_common(30):
#             uniprot_id = self.protein_uniprot_mapping.get(protein, {}).get('accession', 'Unknown')
#             protein_name = self.protein_uniprot_mapping.get(protein, {}).get('protein_name', 'Unknown')

#             # Find associated side effects
#             associated_effects = []
#             for doc_id, proteins in self.protein_mentions.items():
#                 if protein in proteins and doc_id in self.side_effect_mentions:
#                     associated_effects.extend(self.side_effect_mentions[doc_id])

#             top_effects = [effect for effect, _ in Counter(associated_effects).most_common(3)]

#             # Check extraction method
#             regex_count = 0
#             llm_count = 0

#             for doc_id, proteins in self.protein_mentions.items():
#                 if protein in proteins:
#                     if doc_id in self.regex_results and 'proteins' in self.regex_results[doc_id]:
#                         if protein in self.regex_results[doc_id]['proteins']:
#                             regex_count += 1
#                     if doc_id in self.llm_results and 'proteins' in self.llm_results[doc_id]:
#                         if protein in self.llm_results[doc_id]['proteins']:
#                             llm_count += 1

#             analysis_data.append({
#                 'Protein': protein,
#                 'UniProt_ID': uniprot_id,
#                 'Protein_Name': protein_name,
#                 'Total_Mentions': count,
#                 'Regex_Mentions': regex_count,
#                 'LLM_Mentions': llm_count,
#                 'Top_Associated_Side_Effects': ', '.join(top_effects)
#             })

#         df = pd.DataFrame(analysis_data)
#         return df

#     def compare_extraction_methods(self) -> Dict:
#         """
#         Compare regex and LLM extraction methods.

#         Returns:
#             Dictionary with comparison statistics
#         """
#         print("Comparing extraction methods...")

#         # Get all proteins found by each method
#         regex_proteins = set()
#         llm_proteins = set()

#         for doc_id, results in self.regex_results.items():
#             if 'proteins' in results:
#                 regex_proteins.update(results['proteins'])

#         for doc_id, results in self.llm_results.items():
#             if 'proteins' in results:
#                 llm_proteins.update(results['proteins'])

#         # Calculate overlap
#         overlap = regex_proteins.intersection(llm_proteins)
#         regex_only = regex_proteins - llm_proteins
#         llm_only = llm_proteins - regex_proteins

#         # Get side effects
#         regex_side_effects = set()
#         llm_side_effects = set()

#         for doc_id, results in self.regex_results.items():
#             if 'side_effects' in results:
#                 regex_side_effects.update(results['side_effects'])

#         for doc_id, results in self.llm_results.items():
#             if 'side_effects' in results:
#                 llm_side_effects.update(results['side_effects'])

#         side_effect_overlap = regex_side_effects.intersection(llm_side_effects)
#         regex_se_only = regex_side_effects - llm_side_effects
#         llm_se_only = llm_side_effects - regex_side_effects

#         comparison = {
#             'proteins': {
#                 'regex_total': len(regex_proteins),
#                 'llm_total': len(llm_proteins),
#                 'overlap': len(overlap),
#                 'regex_only': len(regex_only),
#                 'llm_only': len(llm_only),
#                 'overlap_proteins': list(overlap),
#                 'regex_only_proteins': list(regex_only),
#                 'llm_only_proteins': list(llm_only)
#             },
#             'side_effects': {
#                 'regex_total': len(regex_side_effects),
#                 'llm_total': len(llm_side_effects),
#                 'overlap': len(side_effect_overlap),
#                 'regex_only': len(regex_se_only),
#                 'llm_only': len(llm_se_only),
#                 'overlap_side_effects': list(side_effect_overlap),
#                 'regex_only_side_effects': list(regex_se_only),
#                 'llm_only_side_effects': list(llm_se_only)
#             }
#         }

#         return comparison

#     def visualize_method_comparison(self, comparison: Dict):
#         """
#         Visualize the comparison between extraction methods.

#         Args:
#             comparison: Comparison dictionary from compare_extraction_methods()
#         """
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

#         # Protein extraction comparison
#         protein_data = comparison['proteins']
#         labels = ['Regex Only', 'LLM Only', 'Both Methods']
#         sizes = [protein_data['regex_only'], protein_data['llm_only'], protein_data['overlap']]
#         colors = ['lightcoral', 'lightblue', 'lightgreen']

#         ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#         ax1.set_title('Protein Extraction Method Comparison')

#         # Side effects comparison
#         se_data = comparison['side_effects']
#         sizes_se = [se_data['regex_only'], se_data['llm_only'], se_data['overlap']]

#         ax2.pie(sizes_se, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
#         ax2.set_title('Side Effects Extraction Method Comparison')

#         # Bar chart for total counts
#         methods = ['Regex', 'LLM', 'Combined']
#         protein_counts = [protein_data['regex_total'], protein_data['llm_total'],
#                          len(set(comparison['proteins']['overlap_proteins'] +
#                                 comparison['proteins']['regex_only_proteins'] +
#                                 comparison['proteins']['llm_only_proteins']))]

#         ax3.bar(methods, protein_counts, color=['lightcoral', 'lightblue', 'lightgreen'])
#         ax3.set_title('Total Unique Proteins Found')
#         ax3.set_ylabel('Number of Proteins')

#         # Bar chart for side effects
#         se_counts = [se_data['regex_total'], se_data['llm_total'],
#                     len(set(comparison['side_effects']['overlap_side_effects'] +
#                            comparison['side_effects']['regex_only_side_effects'] +
#                            comparison['side_effects']['llm_only_side_effects']))]

#         ax4.bar(methods, se_counts, color=['lightcoral', 'lightblue', 'lightgreen'])
#         ax4.set_title('Total Unique Side Effects Found')
#         ax4.set_ylabel('Number of Side Effects')

#         plt.tight_layout()
#         plt.show()

#     def run_corpus_analysis(self, corpus_path: str, corpus_type: str = "auto") -> Dict:
#         """
#         Run analysis on a literature corpus using dual extraction methods.

#         Args:
#             corpus_path: Path to literature corpus
#             corpus_type: Type of corpus ('txt', 'json', 'csv', 'pdf', 'auto')

#         Returns:
#             Dictionary containing all results
#         """
#         print("=== Starting Literature Corpus Analysis ===")

#         # Load corpus
#         documents = self.process_literature_corpus(corpus_path, corpus_type)
#         if not documents:
#             print("No documents found in corpus")
#             return {}

#         # Extract proteins and side effects using dual approach
#         self.extract_proteins_and_side_effects_dual(documents)

#         # Get all unique proteins
#         all_proteins = set()
#         for proteins in self.protein_mentions.values():
#             all_proteins.update(proteins)

#         if not all_proteins:
#             print("No proteins found")
#             return {}

#         # Map to UniProt
#         uniprot_mapping = self.map_to_uniprot(all_proteins)

#         # Get StringDB network
#         uniprot_ids = list(uniprot_mapping.values())
#         if uniprot_ids:
#             network_data = self.get_string_network(uniprot_ids[:50])  # Limit to 50 for demo
#             if network_data:
#                 self.string_networks = network_data

#         # Analyze results
#         analysis_df = self.analyze_results()

#         # Compare methods
#         method_comparison = self.compare_extraction_methods()

#         # Create network graph
#         network_graph = self.create_network_graph(self.string_networks)

#         # Visualize comparisons
#         if self.use_llm:
#             self.visualize_method_comparison(method_comparison)

#         # Visualize network if it exists
#         if len(network_graph.nodes()) > 0:
#             self.visualize_network(network_graph)

#         results = {
#             'documents': documents,
#             'protein_mentions': dict(self.protein_mentions),
#             'side_effect_mentions': dict(self.side_effect_mentions),
#             'regex_results': dict(self.regex_results),
#             'llm_results': dict(self.llm_results),
#             'protein_side_effect_links': dict(self.protein_side_effect_links),
#             'uniprot_mapping': uniprot_mapping,
#             'string_networks': self.string_networks,
#             'analysis_df': analysis_df,
#             'method_comparison': method_comparison,
#             'network_graph': network_graph
#         }

#         print("=== Analysis Complete ===")
#         print(f"Processed {len(documents)} documents")
#         print(f"Found {len(all_proteins)} unique proteins")
#         print(f"Mapped {len(uniprot_mapping)} proteins to UniProt")
#         print(f"Network contains {len(network_graph.nodes())} nodes and {len(network_graph.edges())} edges")

#         if self.use_llm:
#             print(f"Regex method found {method_comparison['proteins']['regex_total']} proteins")
#             print(f"LLM method found {method_comparison['proteins']['llm_total']} proteins")
#             print(f"Method overlap: {method_comparison['proteins']['overlap']} proteins")

#         return results

#     def run_analysis(self, query: str, max_results: int = 100) -> Dict:
#         """
#         Run the complete analysis pipeline.

#         Args:
#             query: PubMed search query
#             max_results: Maximum number of articles to process

#         Returns:
#             Dictionary containing all results
#         """
#         print("=== Starting PubMed Protein Side Effects Analysis ===")

#         # Search PubMed
#         pmids = self.search_pubmed(query, max_results)
#         if not pmids:
#             print("No articles found")
#             return {}

#         # Fetch abstracts
#         abstracts = self.fetch_abstracts(pmids)
#         if not abstracts:
#             print("No abstracts retrieved")
#             return {}

#         # Extract proteins and side effects
#         self.extract_proteins_and_side_effects(abstracts)

#         # Get all unique proteins
#         all_proteins = set()
#         for proteins in self.protein_mentions.values():
#             all_proteins.update(proteins)

#         if not all_proteins:
#             print("No proteins found")
#             return {}

#         # Map to UniProt
#         uniprot_mapping = self.map_to_uniprot(all_proteins)

#         # Get StringDB network
#         uniprot_ids = list(uniprot_mapping.values())
#         if uniprot_ids:
#             network_data = self.get_string_network(uniprot_ids[:50])  # Limit to 50 for demo
#             if network_data:
#                 self.string_networks = network_data

#         # Analyze results
#         analysis_df = self.analyze_results()

#         # Create network graph
#         network_graph = self.create_network_graph(self.string_networks)

#         # Visualize if network exists
#         if len(network_graph.nodes()) > 0:
#             self.visualize_network(network_graph)

#         results = {
#             'abstracts': abstracts,
#             'protein_mentions': dict(self.protein_mentions),
#             'side_effect_mentions': dict(self.side_effect_mentions),
#             'uniprot_mapping': uniprot_mapping,
#             'string_networks': self.string_networks,
#             'analysis_df': analysis_df,
#             'network_graph': network_graph
#         }

#         print("=== Analysis Complete ===")
#         print(f"Processed {len(abstracts)} abstracts")
#         print(f"Found {len(all_proteins)} unique proteins")
#         print(f"Mapped {len(uniprot_mapping)} proteins to UniProt")
#         print(f"Network contains {len(network_graph.nodes())} nodes and {len(network_graph.edges())} edges")

#         return results

# # Example usage
# if __name__ == "__main__":
#     # Initialize the miner
#     miner = PubMedProteinMiner(
#         email="your_email@example.com",
#         use_llm=True,  # Enable LLM extraction
#         ollama_model="llama3.1:8b"  # Specify model
#     )

#     # === Example 1: PubMed Analysis ===
#     print("=== PubMed Analysis Example ===")
#     pubmed_queries = [
#         "drug induced liver injury proteins",
#         "cardiotoxicity protein biomarkers"
#     ]

#     for query in pubmed_queries[:1]:  # Run one query for demo
#         print(f"\nAnalyzing PubMed query: {query}")
#         results = miner.run_analysis(query, max_results=50)

#         if results and 'analysis_df' in results:
#             print("\nTop proteins associated with side effects:")
#             print(results['analysis_df'].head(10))

#             # Save results
#             results['analysis_df'].to_csv(f"pubmed_{query.replace(' ', '_')}_analysis.csv", index=False)

#     # === Example 2: Literature Corpus Analysis ===
#     print("\n=== Literature Corpus Analysis Example ===")

#     # Example corpus paths (adjust these to your actual data)
#     corpus_examples = [
#         # Text files directory
#         {
#             'path': '/path/to/literature/txt_files/',
#             'type': 'txt',
#             'description': 'Directory with text files'
#         },
#         # JSON corpus file
#         {
#             'path': '/path/to/literature/corpus.json',
#             'type': 'json',
#             'description': 'JSON corpus file'
#         },
#         # CSV corpus file
#         {
#             'path': '/path/to/literature/papers.csv',
#             'type': 'csv',
#             'description': 'CSV file with papers'
#         },
#         # PDF directory
#         {
#             'path': '/path/to/literature/pdf_files/',
#             'type': 'pdf',
#             'description': 'Directory with PDF files'
#         }
#     ]

#     # Example: Process text files
#     # Uncomment and modify the path to use:
#     """
#     corpus_path = "/path/to/your/literature/corpus"
#     corpus_results = miner.run_corpus_analysis(corpus_path, corpus_type="auto")

#     if corpus_results and 'analysis_df' in corpus_results:
#         print("\nCorpus Analysis Results:")
#         print(corpus_results['analysis_df'].head(15))

#         # Save comprehensive results
#         corpus_results['analysis_df'].to_csv("corpus_analysis_results.csv", index=False)

#         # Save method comparison
#         if 'method_comparison' in corpus_results:
#             comparison_df = pd.DataFrame([
#                 {
#                     'Method': 'Regex',
#                     'Proteins_Found': corpus_results['method_comparison']['proteins']['regex_total'],
#                     'Side_Effects_Found': corpus_results['method_comparison']['side_effects']['regex_total']
#                 },
#                 {
#                     'Method': 'LLM',
#                     'Proteins_Found': corpus_results['method_comparison']['proteins']['llm_total'],
#                     'Side_Effects_Found': corpus_results['method_comparison']['side_effects']['llm_total']
#                 },
#                 {
#                     'Method': 'Overlap',
#                     'Proteins_Found': corpus_results['method_comparison']['proteins']['overlap'],
#                     'Side_Effects_Found': corpus_results['method_comparison']['side_effects']['overlap']
#                 }
#             ])
#             comparison_df.to_csv("method_comparison.csv", index=False)
#     """

#     # === Example 3: Create sample corpus for testing ===
#     print("\n=== Creating Sample Corpus for Testing ===")

#     # Create sample documents
#     sample_docs = [
#         {
#             'id': 'doc1',
#             'title': 'Hepatotoxicity of Paracetamol',
#             'content': 'Paracetamol overdose causes severe hepatotoxicity through CYP2E1 activation. The protein p53 is involved in liver cell apoptosis. TNF-α and IL-6 mediate inflammatory responses leading to liver damage.'
#         },
#         {
#             'id': 'doc2',
#             'title': 'Cardiotoxicity of Doxorubicin',
#             'content': 'Doxorubicin-induced cardiotoxicity involves topoisomerase II inhibition. VEGF signaling is disrupted, leading to cardiac dysfunction. NF-κB activation promotes inflammatory pathways causing heart failure.'
#         },
#         {
#             'id': 'doc3',
#             'title': 'Nephrotoxicity Mechanisms',
#             'content': 'Cisplatin nephrotoxicity involves DNA damage and apoptosis. EGFR and MAPK pathways are activated in response to kidney injury. Cytochrome P450 enzymes contribute to toxic metabolite formation.'
#         }
#     ]

#     # Save sample corpus as JSON
#     with open('sample_corpus.json', 'w') as f:
#         json.dump(sample_docs, f, indent=2)

#     print("Sample corpus created as 'sample_corpus.json'")

#     # Process sample corpus
#     print("\nProcessing sample corpus...")
#     sample_results = miner.run_corpus_analysis('sample_corpus.json', corpus_type='json')

#     if sample_results and 'analysis_df' in sample_results:
#         print("\nSample Corpus Results:")
#         print(sample_results['analysis_df'])

#         if 'method_comparison' in sample_results:
#             print("\nMethod Comparison:")
#             comp = sample_results['method_comparison']
#             print(f"Regex found {comp['proteins']['regex_total']} proteins")
#             print(f"LLM found {comp['proteins']['llm_total']} proteins")
#             print(f"Overlap: {comp['proteins']['overlap']} proteins")

#     print("\n=== Usage Instructions ===")
#     print("""
#     To use this tool with your own data:

#     1. Install Ollama:
#        - Visit https://ollama.ai/
#        - Install Ollama on your system
#        - Pull a model: ollama pull llama3.1:8b

#     2. Install Python dependencies:
#        pip install biopython requests pandas networkx matplotlib seaborn nltk spacy PyPDF2
#        python -m spacy download en_core_web_sm

#     3. Prepare your literature corpus:
#        - Text files: Place .txt files in a directory
#        - JSON: Create a JSON file with document objects
#        - CSV: Create a CSV with 'text', 'id', and 'title' columns
#        - PDF: Place .pdf files in a directory

#     4. Run analysis:
#        miner = PubMedProteinMiner(email="your_email@example.com", use_llm=True)
#        results = miner.run_corpus_analysis("/path/to/your/corpus", corpus_type="auto")

#     5. Access results:
#        - results['analysis_df']: Main analysis DataFrame
#        - results['method_comparison']: Comparison of regex vs LLM methods
#        - results['protein_side_effect_links']: LLM-extracted relationships
#        - results['network_graph']: Protein interaction network
#     """)
# , ent.text):
#                             regex_proteins.add(ent.text)

#             # Filter proteins
#             filtered_regex_proteins = self._filter_proteins(regex_proteins)

#             # Store regex results
#             if filtered_regex_proteins:
#                 self.regex_results[doc_id]['proteins'] = filtered_regex_proteins
#             if regex_side_effects:
#                 self.regex_results[doc_id]['side_effects'] = regex_side_effects

#             # === LLM-based extraction ===
#             if self.use_llm:
#                 # Split text into chunks for LLM processing
#                 chunks = self._split_text_into_chunks(full_text, max_chunk_size=2000)

#                 llm_proteins = set()
#                 llm_side_effects = set()
#                 llm_links = []

#                 for chunk in chunks:
#                     # Extract proteins
#                     chunk_proteins = self.llm_extractor.extract_proteins_llm(chunk)
#                     llm_proteins.update(chunk_proteins)

#                     # Extract side effects
#                     chunk_side_effects = self.llm_extractor.extract_side_effects_llm(chunk)
#                     llm_side_effects.update(chunk_side_effects)

#                     # Extract relationships
#                     chunk_links = self.llm_extractor.extract_protein_side_effect_links_llm(chunk)
#                     llm_links.extend(chunk_links)

#                 # Store LLM results
#                 if llm_proteins:
#                     self.llm_results[doc_id]['proteins'] = list(llm_proteins)
#                 if llm_side_effects:
#                     self.llm_results[doc_id]['side_effects'] = list(llm_side_effects)
#                 if llm_links:
#                     self.protein_side_effect_links[doc_id] = llm_links

#             # === Combine results ===
#             all_proteins = set(filtered_regex_proteins)
#             all_side_effects = set(regex_side_effects)

#             if doc_id in self.llm_results:
#                 if 'proteins' in self.llm_results[doc_id]:
#                     all_proteins.update(self.llm_results[doc_id]['proteins'])
#                 if 'side_effects' in self.llm_results[doc_id]:
#                     all_side_effects.update(self.llm_results[doc_id]['side_effects'])

#             # Store combined results
#             if all_proteins:
#                 self.protein_mentions[doc_id] = list(all_proteins)
#             if all_side_effects:
#                 self.side_effect_mentions[doc_id] = list(all_side_effects)

#         print(f"Regex extraction found proteins in {len(self.regex_results)} documents")
#         if self.use_llm:
#             print(f"LLM extraction found proteins in {len(self.llm_results)} documents")
#         print(f"Combined results: proteins in {len(self.protein_mentions)} documents")

#     def _split_text_into_chunks(self, text: str, max_chunk_size: int = 2000) -> List[str]:
#         """
#         Split text into chunks for LLM processing.

#         Args:
#             text: Input text
#             max_chunk_size: Maximum chunk size in characters

#         Returns:
#             List of text chunks
#         """
#         if len(text) <= max_chunk_size:
#             return [text]

#         # Split by sentences first
#         sentences = sent_tokenize(text)
#         chunks = []
#         current_chunk = ""

#         for sentence in sentences:
#             if len(current_chunk) + len(sentence) <= max_chunk_size:
#                 current_chunk += " " + sentence
#             else:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = sentence

#         if current_chunk:
#             chunks.append(current_chunk.strip())

#         return chunks

#     def _filter_proteins(self, proteins: Set[str]) -> List[str]:
#         """
#         Filter out common false positives from protein mentions.

#         Args:
#             proteins: Set of protein names

#         Returns:
#             Filtered list of proteins
#         """
#         filtered_proteins = []
#         stop_words = set(stopwords.words('english'))

#         # Common false positives in biomedical text
#         false_positives = {
#             'RNA', 'DNA', 'ATP', 'ADP', 'GTP', 'GDP', 'NAD', 'NADH', 'FAD', 'FADH',
#             'USA', 'UK', 'EU', 'US', 'FDA', 'WHO', 'NIH', 'CDC', 'DMSO', 'PBS',
#             'Fig', 'Figure', 'Table', 'Ref', 'et', 'al', 'vs', 'Inc', 'Ltd', 'Co',
#             'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD'
#         }

#         for protein in proteins:
#             if (len(protein) > 2 and
#                 protein.lower() not in stop_words and
#                 protein not in false_positives and
#                 not protein.lower() in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'] and
#                 not protein.isdigit() and
#                 not all(c.isdigit() or c in '.,()-' for c in protein)):
#                 filtered_proteins.append(protein)

#         return filtered_proteins#!/usr/bin/env python3
# """
# PubMed Protein Side Effects Text Miner
# ======================================

# This tool mines PubMed abstracts for proteins related to side effects,
# maps them to UniProt identifiers, and extracts protein networks using StringDB.

# Dependencies:
# pip install biopython requests pandas networkx matplotlib seaborn nltk spacy

# You'll also need to download the spaCy model:
# python -m spacy download en_core_web_sm

# Author: Claude
# """

# import requests
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# import seaborn as sns
# from Bio import Entrez
# import re
# import json
# import time
# import os
# import glob
# from pathlib import Path
# from collections import defaultdict, Counter
# from typing import List, Dict, Set, Tuple, Optional, Union
# import warnings
# warnings.filterwarnings('ignore')

# # NLP imports
# import nltk
# import spacy
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')

# class OllamaLLMExtractor:
#     """
#     Local LLM-based protein and side effect extractor using Ollama.
#     """

#     def __init__(self, model_name: str = "llama3.1:8b", ollama_host: str = "http://localhost:11434"):
#         """
#         Initialize the Ollama LLM extractor.

#         Args:
#             model_name: Name of the Ollama model to use
#             ollama_host: Ollama server host URL
#         """
#         self.model_name = model_name
#         self.ollama_host = ollama_host
#         self.available_models = self._get_available_models()

#         # Extraction prompts
#         self.protein_extraction_prompt = """
# You are a biomedical text mining expert. Extract all protein names, gene names, and enzyme names from the following text.

# Important instructions:
# 1. Include common protein names (e.g., p53, TNF-α, VEGF)
# 2. Include gene names (e.g., BRCA1, EGFR, MYC)
# 3. Include enzyme names (e.g., cytochrome P450, kinase names)
# 4. Include protein complexes (e.g., NF-κB, AP-1)
# 5. Only extract names that clearly refer to proteins/genes/enzymes
# 6. Do not extract drug names, disease names, or general biological terms
# 7. Return results as a JSON list of unique protein names

# Text to analyze:
# {text}

# Return only the JSON list, no other text:
# """

#         self.side_effect_extraction_prompt = """
# You are a medical text mining expert. Extract all adverse effects, side effects, toxicities, and harmful reactions mentioned in the following text.

# Important instructions:
# 1. Include specific adverse effects (e.g., hepatotoxicity, nephrotoxicity)
# 2. Include general side effects (e.g., nausea, headache, rash)
# 3. Include toxic effects (e.g., cardiotoxicity, neurotoxicity)
# 4. Include allergic reactions and hypersensitivity
# 5. Include organ damage or dysfunction
# 6. Normalize similar terms (e.g., "liver toxicity" → "hepatotoxicity")
# 7. Return results as a JSON list of unique side effect terms

# Text to analyze:
# {text}

# Return only the JSON list, no other text:
# """

#         self.protein_side_effect_link_prompt = """
# You are a biomedical expert. Analyze the following text and identify relationships between proteins and side effects/adverse events.

# Extract information in this format:
# - Protein name
# - Associated side effect
# - Relationship type (causes, associated_with, biomarker_for, protective_against)
# - Confidence (high, medium, low)

# Text to analyze:
# {text}

# Return results as JSON in this format:
# [
#   {{
#     "protein": "protein_name",
#     "side_effect": "side_effect_name",
#     "relationship": "relationship_type",
#     "confidence": "confidence_level"
#   }}
# ]

# Return only the JSON, no other text:
# """

#     def _get_available_models(self) -> List[str]:
#         """Get list of available Ollama models."""
#         try:
#             response = requests.get(f"{self.ollama_host}/api/tags")
#             if response.status_code == 200:
#                 models = response.json()
#                 return [model['name'] for model in models['models']]
#             return []
#         except Exception as e:
#             print(f"Warning: Could not connect to Ollama server: {e}")
#             return []

#     def _query_ollama(self, prompt: str, max_retries: int = 3) -> str:
#         """
#         Query Ollama API with retry logic.

#         Args:
#             prompt: The prompt to send to the model
#             max_retries: Maximum number of retries

#         Returns:
#             Model response text
#         """
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     f"{self.ollama_host}/api/generate",
#                     json={
#                         "model": self.model_name,
#                         "prompt": prompt,
#                         "stream": False,
#                         "options": {
#                             "temperature": 0.1,  # Low temperature for consistent extraction
#                             "top_p": 0.9,
#                             "num_predict": 2000
#                         }
#                     },
#                     timeout=120
#                 )

#                 if response.status_code == 200:
#                     return response.json()['response']
#                 else:
#                     print(f"Ollama API error: {response.status_code}")

#             except Exception as e:
#                 print(f"Ollama query attempt {attempt + 1} failed: {e}")
#                 if attempt < max_retries - 1:
#                     time.sleep(2 ** attempt)  # Exponential backoff

#         return ""

#     def extract_proteins_llm(self, text: str) -> List[str]:
#         """
#         Extract proteins using LLM.

#         Args:
#             text: Input text to analyze

#         Returns:
#             List of extracted protein names
#         """
#         prompt = self.protein_extraction_prompt.format(text=text)
#         response = self._query_ollama(prompt)

#         try:
#             # Try to parse JSON response
#             proteins = json.loads(response.strip())
#             if isinstance(proteins, list):
#                 return [p.strip() for p in proteins if p.strip()]
#             return []
#         except json.JSONDecodeError:
#             # Fallback: extract from text response
#             proteins = []
#             lines = response.split('\n')
#             for line in lines:
#                 line = line.strip()
#                 if line.startswith('-') or line.startswith('•'):
#                     protein = line[1:].strip()
#                     if protein:
#                         proteins.append(protein)
#             return proteins

#     def extract_side_effects_llm(self, text: str) -> List[str]:
#         """
#         Extract side effects using LLM.

#         Args:
#             text: Input text to analyze

#         Returns:
#             List of extracted side effects
#         """
#         prompt = self.side_effect_extraction_prompt.format(text=text)
#         response = self._query_ollama(prompt)

#         try:
#             side_effects = json.loads(response.strip())
#             if isinstance(side_effects, list):
#                 return [s.strip() for s in side_effects if s.strip()]
#             return []
#         except json.JSONDecodeError:
#             side_effects = []
#             lines = response.split('\n')
#             for line in lines:
#                 line = line.strip()
#                 if line.startswith('-') or line.startswith('•'):
#                     effect = line[1:].strip()
#                     if effect:
#                         side_effects.append(effect)
#             return side_effects

#     def extract_protein_side_effect_links_llm(self, text: str) -> List[Dict]:
#         """
#         Extract protein-side effect relationships using LLM.

#         Args:
#             text: Input text to analyze

#         Returns:
#             List of relationship dictionaries
#         """
#         prompt = self.protein_side_effect_link_prompt.format(text=text)
#         response = self._query_ollama(prompt)

#         try:
#             relationships = json.loads(response.strip())
#             if isinstance(relationships, list):
#                 return relationships
#             return []
#         except json.JSONDecodeError:
#             return []

# class LiteratureCorpusProcessor:
#     """
#     Process various literature formats (PDF, TXT, JSON, etc.) for text mining.
#     """

#     def __init__(self):
#         """Initialize the corpus processor."""
#         self.supported_formats = ['.txt', '.json', '.csv', '.pdf']

#     def read_text_files(self, directory: str, pattern: str = "*.txt") -> List[Dict]:
#         """
#         Read text files from a directory.

#         Args:
#             directory: Directory containing text files
#             pattern: File pattern to match

#         Returns:
#             List of document dictionaries
#         """
#         documents = []
#         files = glob.glob(os.path.join(directory, pattern))

#         for file_path in files:
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     documents.append({
#                         'id': os.path.basename(file_path),
#                         'source': file_path,
#                         'content': content,
#                         'title': os.path.basename(file_path)
#                     })
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")

#         return documents

#     def read_json_corpus(self, file_path: str) -> List[Dict]:
#         """
#         Read JSON corpus file.

#         Args:
#             file_path: Path to JSON file

#         Returns:
#             List of document dictionaries
#         """
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             if isinstance(data, list):
#                 return data
#             elif isinstance(data, dict):
#                 return [data]
#             return []
#         except Exception as e:
#             print(f"Error reading JSON corpus: {e}")
#             return []

#     def read_csv_corpus(self, file_path: str, text_column: str = 'text',
#                        id_column: str = 'id', title_column: str = 'title') -> List[Dict]:
#         """
#         Read CSV corpus file.

#         Args:
#             file_path: Path to CSV file
#             text_column: Column name containing text
#             id_column: Column name containing document ID
#             title_column: Column name containing title

#         Returns:
#             List of document dictionaries
#         """
#         try:
#             df = pd.read_csv(file_path)
#             documents = []

#             for _, row in df.iterrows():
#                 doc = {
#                     'id': row.get(id_column, f"doc_{len(documents)}"),
#                     'source': file_path,
#                     'content': row.get(text_column, ''),
#                     'title': row.get(title_column, row.get(id_column, ''))
#                 }
#                 documents.append(doc)

#             return documents
#         except Exception as e:
#             print(f"Error reading CSV corpus: {e}")
#             return []

#     def process_pdf_directory(self, directory: str) -> List[Dict]:
#         """
#         Process PDF files in a directory (requires PyPDF2).

#         Args:
#             directory: Directory containing PDF files

#         Returns:
#             List of document dictionaries
#         """
#         documents = []

#         try:
#             import PyPDF2

#             pdf_files = glob.glob(os.path.join(directory, "*.pdf"))

#             for pdf_path in pdf_files:
#                 try:
#                     with open(pdf_path, 'rb') as file:
#                         pdf_reader = PyPDF2.PdfReader(file)
#                         text = ""

#                         for page in pdf_reader.pages:
#                             text += page.extract_text()

#                         documents.append({
#                             'id': os.path.basename(pdf_path),
#                             'source': pdf_path,
#                             'content': text,
#                             'title': os.path.basename(pdf_path)
#                         })

#                 except Exception as e:
#                     print(f"Error processing PDF {pdf_path}: {e}")

#         except ImportError:
#             print("PyPDF2 not installed. Install with: pip install PyPDF2")

#         return documents
#     """
#     A comprehensive text mining tool for extracting proteins related to side effects
#     from PubMed abstracts with UniProt mapping and StringDB network analysis.
#     """

#     def __init__(self, email: str, max_retries: int = 3):
#         """
#         Initialize the miner with configuration.

#         Args:
#             email: Email for NCBI API access (required)
#             max_retries: Maximum number of retries for API calls
#         """
#         self.email = email
#         self.max_retries = max_retries
#         Entrez.email = email

#         # Load spaCy model for NER
#         try:
#             self.nlp = spacy.load("en_core_web_sm")
#         except OSError:
#             print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
#             self.nlp = None

#         # Initialize data storage
#         self.abstracts = []
#         self.protein_mentions = defaultdict(list)
#         self.side_effect_mentions = defaultdict(list)
#         self.protein_uniprot_mapping = {}
#         self.string_networks = {}

#         # Define side effect keywords
#         self.side_effect_keywords = {
#             'adverse effects', 'side effects', 'toxicity', 'adverse reactions',
#             'adverse events', 'hepatotoxicity', 'nephrotoxicity', 'cardiotoxicity',
#             'neurotoxicity', 'cytotoxicity', 'genotoxicity', 'mutagenicity',
#             'carcinogenicity', 'teratogenicity', 'immunotoxicity', 'allergic reaction',
#             'hypersensitivity', 'anaphylaxis', 'rash', 'nausea', 'vomiting',
#             'diarrhea', 'headache', 'dizziness', 'fatigue', 'insomnia',
#             'depression', 'anxiety', 'seizure', 'tremor', 'muscle pain',
#             'joint pain', 'fever', 'chills', 'weight gain', 'weight loss',
#             'hair loss', 'skin reactions', 'liver damage', 'kidney damage',
#             'heart problems', 'blood disorders', 'bone marrow suppression'
#         }

#         # Common protein name patterns
#         self.protein_patterns = [
#             r'\b[A-Z][a-z]+\d+[a-z]*\b',  # e.g., Cyp3a4, Tnf1a
#             r'\b[A-Z]{2,}[0-9]*[A-Z]*\b',  # e.g., TNF, IL1B, VEGFA
#             r'\b[A-Z][a-z]+-?[A-Z0-9]+\b',  # e.g., p53, NF-kB
#             r'\b[A-Z]{1,2}[0-9]+[A-Z]*\b',  # e.g., P53, IL6
#         ]

#     def search_pubmed(self, query: str, max_results: int = 100) -> List[str]:
#         """
#         Search PubMed for articles matching the query.

#         Args:
#             query: Search query string
#             max_results: Maximum number of results to retrieve

#         Returns:
#             List of PubMed IDs
#         """
#         print(f"Searching PubMed for: {query}")

#         try:
#             # Search for articles
#             search_handle = Entrez.esearch(
#                 db="pubmed",
#                 term=query,
#                 retmax=max_results,
#                 sort="pub_date"
#             )
#             search_results = Entrez.read(search_handle)
#             search_handle.close()

#             pmids = search_results["IdList"]
#             print(f"Found {len(pmids)} articles")
#             return pmids

#         except Exception as e:
#             print(f"Error searching PubMed: {e}")
#             return []

#     def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
#         """
#         Fetch abstracts for given PubMed IDs.

#         Args:
#             pmids: List of PubMed IDs

#         Returns:
#             List of dictionaries containing abstract information
#         """
#         print(f"Fetching abstracts for {len(pmids)} articles...")
#         abstracts = []

#         # Process in batches to avoid overwhelming the API
#         batch_size = 20
#         for i in range(0, len(pmids), batch_size):
#             batch = pmids[i:i + batch_size]

#             try:
#                 # Fetch abstracts
#                 fetch_handle = Entrez.efetch(
#                     db="pubmed",
#                     id=",".join(batch),
#                     rettype="abstract",
#                     retmode="xml"
#                 )
#                 fetch_results = Entrez.read(fetch_handle)
#                 fetch_handle.close()

#                 # Parse results
#                 for article in fetch_results["PubmedArticle"]:
#                     try:
#                         medline_citation = article["MedlineCitation"]
#                         pmid = str(medline_citation["PMID"])

#                         # Extract title
#                         title = medline_citation["Article"]["ArticleTitle"]

#                         # Extract abstract
#                         abstract_text = ""
#                         if "Abstract" in medline_citation["Article"]:
#                             abstract_sections = medline_citation["Article"]["Abstract"]["AbstractText"]
#                             if isinstance(abstract_sections, list):
#                                 abstract_text = " ".join([str(section) for section in abstract_sections])
#                             else:
#                                 abstract_text = str(abstract_sections)

#                         # Extract publication date
#                         pub_date = "Unknown"
#                         if "PubDate" in medline_citation["Article"]["Journal"]["JournalIssue"]:
#                             date_info = medline_citation["Article"]["Journal"]["JournalIssue"]["PubDate"]
#                             if "Year" in date_info:
#                                 pub_date = str(date_info["Year"])

#                         abstracts.append({
#                             "pmid": pmid,
#                             "title": title,
#                             "abstract": abstract_text,
#                             "pub_date": pub_date
#                         })

#                     except Exception as e:
#                         print(f"Error processing article: {e}")
#                         continue

#                 # Be respectful to NCBI servers
#                 time.sleep(0.5)

#             except Exception as e:
#                 print(f"Error fetching batch: {e}")
#                 continue

#         print(f"Successfully fetched {len(abstracts)} abstracts")
#         return abstracts

#     def extract_proteins_and_side_effects(self, abstracts: List[Dict]) -> None:
#         """
#         Extract protein mentions and side effects from abstracts.

#         Args:
#             abstracts: List of abstract dictionaries
#         """
#         print("Extracting proteins and side effects...")

#         for abstract in abstracts:
#             pmid = abstract["pmid"]
#             text = f"{abstract['title']} {abstract['abstract']}".lower()

#             # Extract side effects
#             side_effects = []
#             for keyword in self.side_effect_keywords:
#                 if keyword in text:
#                     side_effects.append(keyword)

#             if side_effects:
#                 self.side_effect_mentions[pmid] = side_effects

#                 # Extract proteins using regex patterns
#                 proteins = set()
#                 full_text = f"{abstract['title']} {abstract['abstract']}"

#                 for pattern in self.protein_patterns:
#                     matches = re.findall(pattern, full_text)
#                     proteins.update(matches)

#                 # Use spaCy for additional protein extraction if available
#                 if self.nlp:
#                     doc = self.nlp(full_text)
#                     for ent in doc.ents:
#                         if ent.label_ in ["PERSON", "ORG"] and len(ent.text) < 15:
#                             # Filter for potential protein names
#                             if re.match(r'^[A-Z][a-z]*[0-9]*[A-Z]*$', ent.text):
#                                 proteins.add(ent.text)

#                 # Filter out common false positives
#                 filtered_proteins = []
#                 stop_words = set(stopwords.words('english'))
#                 for protein in proteins:
#                     if (len(protein) > 2 and
#                         protein.lower() not in stop_words and
#                         not protein.lower() in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']):
#                         filtered_proteins.append(protein)

#                 if filtered_proteins:
#                     self.protein_mentions[pmid] = filtered_proteins

#         print(f"Found proteins in {len(self.protein_mentions)} abstracts")
#         print(f"Found side effects in {len(self.side_effect_mentions)} abstracts")

#     def map_to_uniprot(self, proteins: Set[str]) -> Dict[str, str]:
#         """
#         Map protein names to UniProt identifiers.

#         Args:
#             proteins: Set of protein names

#         Returns:
#             Dictionary mapping protein names to UniProt IDs
#         """
#         print(f"Mapping {len(proteins)} proteins to UniProt...")
#         mapping = {}

#         for protein in proteins:
#             try:
#                 # Query UniProt API
#                 url = f"https://rest.uniprot.org/uniprotkb/search"
#                 params = {
#                     'query': f'gene:{protein} OR protein_name:{protein}',
#                     'format': 'json',
#                     'size': 5,
#                     'fields': 'accession,gene_names,protein_name'
#                 }

#                 response = requests.get(url, params=params, timeout=10)

#                 if response.status_code == 200:
#                     data = response.json()
#                     if data['results']:
#                         # Take the first result
#                         result = data['results'][0]
#                         accession = result['primaryAccession']
#                         mapping[protein] = accession

#                         # Store additional info
#                         self.protein_uniprot_mapping[protein] = {
#                             'accession': accession,
#                             'gene_names': result.get('genes', []),
#                             'protein_name': result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown')
#                         }

#                 # Be respectful to the API
#                 time.sleep(0.1)

#             except Exception as e:
#                 print(f"Error mapping {protein}: {e}")
#                 continue

#         print(f"Successfully mapped {len(mapping)} proteins to UniProt")
#         return mapping

#     def get_string_network(self, uniprot_ids: List[str], species: str = "9606") -> Dict:
#         """
#         Retrieve protein-protein interaction network from StringDB.

#         Args:
#             uniprot_ids: List of UniProt identifiers
#             species: Species taxonomy ID (default: 9606 for human)

#         Returns:
#             Dictionary containing network information
#         """
#         print(f"Retrieving StringDB network for {len(uniprot_ids)} proteins...")

#         try:
#             # StringDB API endpoint
#             string_api_url = "https://string-db.org/api"

#             # Get string identifiers
#             url = f"{string_api_url}/json/get_string_ids"
#             params = {
#                 'identifiers': '\r'.join(uniprot_ids),
#                 'species': species,
#                 'limit': 1,
#                 'echo_query': 1
#             }

#             response = requests.post(url, data=params)
#             string_ids = response.json()

#             # Map UniProt to String IDs
#             uniprot_to_string = {}
#             for item in string_ids:
#                 if 'stringId' in item:
#                     uniprot_to_string[item['queryItem']] = item['stringId']

#             if not uniprot_to_string:
#                 print("No String IDs found for the provided proteins")
#                 return {}

#             # Get network interactions (excluding text-mining evidence)
#             url = f"{string_api_url}/json/network"
#             params = {
#                 'identifiers': '\r'.join(uniprot_to_string.values()),
#                 'species': species,
#                 'required_score': 400,  # Medium confidence
#                 'hide_disconnected_nodes': 1,
#                 'network_flavor': 'evidence'
#             }

#             response = requests.post(url, data=params)
#             network_data = response.json()

#             # Filter out text-mining evidence
#             filtered_interactions = []
#             for interaction in network_data:
#                 evidence_types = []
#                 for evidence in ['neighborhood', 'fusion', 'cooccurence', 'coexpression',
#                                'experimental', 'database', 'textmining']:
#                     if interaction.get(evidence, 0) > 0:
#                         evidence_types.append(evidence)

#                 # Exclude interactions based only on text-mining
#                 if evidence_types and 'textmining' in evidence_types:
#                     non_textmining = [e for e in evidence_types if e != 'textmining']
#                     if non_textmining:  # Has non-text-mining evidence
#                         interaction['evidence_types'] = non_textmining
#                         filtered_interactions.append(interaction)
#                 elif evidence_types:  # Has evidence and no text-mining
#                     interaction['evidence_types'] = evidence_types
#                     filtered_interactions.append(interaction)

#             print(f"Found {len(filtered_interactions)} protein-protein interactions")

#             return {
#                 'uniprot_to_string': uniprot_to_string,
#                 'interactions': filtered_interactions
#             }

#         except Exception as e:
#             print(f"Error retrieving StringDB network: {e}")
#             return {}

#     def create_network_graph(self, network_data: Dict) -> nx.Graph:
#         """
#         Create a NetworkX graph from StringDB network data.

#         Args:
#             network_data: Network data from StringDB

#         Returns:
#             NetworkX graph object
#         """
#         G = nx.Graph()

#         if not network_data or 'interactions' not in network_data:
#             return G

#         # Add nodes and edges
#         for interaction in network_data['interactions']:
#             node1 = interaction['preferredName_A']
#             node2 = interaction['preferredName_B']
#             score = interaction['score']
#             evidence = interaction.get('evidence_types', [])

#             # Add nodes
#             G.add_node(node1)
#             G.add_node(node2)

#             # Add edge with attributes
#             G.add_edge(node1, node2, weight=score, evidence=evidence)

#         return G

#     def visualize_network(self, G: nx.Graph, title: str = "Protein-Protein Interaction Network"):
#         """
#         Visualize the protein interaction network.

#         Args:
#             G: NetworkX graph
#             title: Plot title
#         """
#         if len(G.nodes()) == 0:
#             print("No network data to visualize")
#             return

#         plt.figure(figsize=(12, 8))

#         # Layout
#         pos = nx.spring_layout(G, k=3, iterations=50)

#         # Draw network
#         nx.draw_networkx_nodes(G, pos, node_color='lightblue',
#                               node_size=1000, alpha=0.7)
#         nx.draw_networkx_labels(G, pos, font_size=8)

#         # Draw edges with varying thickness based on score
#         edges = G.edges()
#         weights = [G[u][v]['weight'] for u, v in edges]
#         nx.draw_networkx_edges(G, pos, width=[w/200 for w in weights],
#                               alpha=0.5, edge_color='gray')

#         plt.title(title)
#         plt.axis('off')
#         plt.tight_layout()
#         plt.show()

#     def analyze_results(self) -> pd.DataFrame:
#         """
#         Analyze and summarize the mining results.

#         Returns:
#             DataFrame with analysis results
#         """
#         print("Analyzing results...")

#         # Count protein mentions
#         all_proteins = []
#         for pmid, proteins in self.protein_mentions.items():
#             all_proteins.extend(proteins)

#         protein_counts = Counter(all_proteins)

#         # Count side effects
#         all_side_effects = []
#         for pmid, effects in self.side_effect_mentions.items():
#             all_side_effects.extend(effects)

#         side_effect_counts = Counter(all_side_effects)

#         # Create analysis DataFrame
#         analysis_data = []
#         for protein, count in protein_counts.most_common(20):
#             uniprot_id = self.protein_uniprot_mapping.get(protein, {}).get('accession', 'Unknown')
#             protein_name = self.protein_uniprot_mapping.get(protein, {}).get('protein_name', 'Unknown')

#             # Find associated side effects
#             associated_effects = []
#             for pmid, proteins in self.protein_mentions.items():
#                 if protein in proteins and pmid in self.side_effect_mentions:
#                     associated_effects.extend(self.side_effect_mentions[pmid])

#             top_effects = [effect for effect, _ in Counter(associated_effects).most_common(3)]

#             analysis_data.append({
#                 'Protein': protein,
#                 'UniProt_ID': uniprot_id,
#                 'Protein_Name': protein_name,
#                 'Mention_Count': count,
#                 'Top_Associated_Side_Effects': ', '.join(top_effects)
#             })

#         df = pd.DataFrame(analysis_data)
#         return df

#     def run_analysis(self, query: str, max_results: int = 100) -> Dict:
#         """
#         Run the complete analysis pipeline.

#         Args:
#             query: PubMed search query
#             max_results: Maximum number of articles to process

#         Returns:
#             Dictionary containing all results
#         """
#         print("=== Starting PubMed Protein Side Effects Analysis ===")

#         # Search PubMed
#         pmids = self.search_pubmed(query, max_results)
#         if not pmids:
#             print("No articles found")
#             return {}

#         # Fetch abstracts
#         abstracts = self.fetch_abstracts(pmids)
#         if not abstracts:
#             print("No abstracts retrieved")
#             return {}

#         # Extract proteins and side effects
#         self.extract_proteins_and_side_effects(abstracts)

#         # Get all unique proteins
#         all_proteins = set()
#         for proteins in self.protein_mentions.values():
#             all_proteins.update(proteins)

#         if not all_proteins:
#             print("No proteins found")
#             return {}

#         # Map to UniProt
#         uniprot_mapping = self.map_to_uniprot(all_proteins)

#         # Get StringDB network
#         uniprot_ids = list(uniprot_mapping.values())
#         if uniprot_ids:
#             network_data = self.get_string_network(uniprot_ids[:50])  # Limit to 50 for demo
#             if network_data:
#                 self.string_networks = network_data

#         # Analyze results
#         analysis_df = self.analyze_results()

#         # Create network graph
#         network_graph = self.create_network_graph(self.string_networks)

#         # Visualize if network exists
#         if len(network_graph.nodes()) > 0:
#             self.visualize_network(network_graph)

#         results = {
#             'abstracts': abstracts,
#             'protein_mentions': dict(self.protein_mentions),
#             'side_effect_mentions': dict(self.side_effect_mentions),
#             'uniprot_mapping': uniprot_mapping,
#             'string_networks': self.string_networks,
#             'analysis_df': analysis_df,
#             'network_graph': network_graph
#         }

#         print("=== Analysis Complete ===")
#         print(f"Processed {len(abstracts)} abstracts")
#         print(f"Found {len(all_proteins)} unique proteins")
#         print(f"Mapped {len(uniprot_mapping)} proteins to UniProt")
#         print(f"Network contains {len(network_graph.nodes())} nodes and {len(network_graph.edges())} edges")

#         return results


# def main():
#     # Initialize the miner
#     miner = PubMedProteinMiner(email="your_email@example.com")

#     # Example queries
#     queries = [
#         "drug induced liver injury proteins",
#         "cardiotoxicity protein biomarkers",
#         "adverse drug reactions proteomics",
#         "hepatotoxicity molecular mechanisms"
#     ]

#     # Run analysis
#     for query in queries[:1]:  # Run one query for demo
#         print(f"\n{'='*50}")
#         print(f"Analyzing: {query}")
#         print(f"{'='*50}")

#         results = miner.run_analysis(query, max_results=50)

#         if results and 'analysis_df' in results:
#             print("\nTop proteins associated with side effects:")
#             print(results['analysis_df'].head(10))

#             # Save results
#             results['analysis_df'].to_csv(f"{query.replace(' ', '_')}_analysis.csv", index=False)
#             print(f"\nResults saved to {query.replace(' ', '_')}_analysis.csv")


# if __name__ == "__main__":
#     main()
