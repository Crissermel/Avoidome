"""
protein_mining_C.py

- Mines PubMed abstracts for proteins related to toxicity/side effects using regex
- Filters extracted proteins using a UniProt dictionary (Swiss-Prot reviewed human proteins)
- Prints a summary: number of unique proteins, most common proteins, etc.
- Saves filtered results to processed_data/regex_extracted_filtered.json

Requirements: requests, pandas, tqdm
"""
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
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProteinToxicityResult:
    pmid: str
    title: str
    abstract: str
    proteins: List[str]
    filtered_proteins: List[str]
    toxicity_terms: List[str] #query
    confidence_score: float
    extraction_method: str
    publication_date: str

class BasePubMedMiner(ABC):
    def __init__(self, email: str, api_key: str, rate_limit: float = 0.34):
        self.email = email
        self.rate_limit = rate_limit
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.toxicity_terms = [
            "toxicity", "adverse effect", "side effect", "hepatotoxicity", "cardiotoxicity", "nephrotoxicity",
            "neurotoxicity", "cytotoxicity", "drug-induced", "adverse drug reaction", "ADR", "drug toxicity",
            "toxic effect", "adverse event", "safety profile", "contraindication", "drug-related",
            "medication-induced", "therapeutic toxicity"
        ]
        self.protein_patterns = [
            r"\b([A-Z][A-Z0-9]+)\b(?:\s+protein|\s+receptor|\s+enzyme|\s+kinase|\s+channel)",
            r"\b(CYP[0-9][A-Z][0-9]+)\b",
            r"\b(P-gp|MDR1|ABCB1)\b",
            r"\b([A-Z]{2,}[0-9]+[A-Z]*)\b(?=\s+(?:protein|receptor|enzyme|kinase|channel|transporter))",
            r"\b(TNF-α|IL-[0-9]+|IFN-[αβγ])\b",
            r"\b(EGFR|VEGFR|PDGFR|FGFR|IGF1R)\b",
            r"\b(BCL-2|BAX|p53|MDM2|BRCA[12])\b",
            r"\b(GSTP1|GSTM1|GSTT1|NAT[12]|SULT[0-9][A-Z][0-9])\b",
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.protein_patterns]

    def _make_request(self, url: str, params: Dict) -> requests.Response:
        params["email"] = self.email
        params["api_key"] = self.api_key
        time.sleep(self.rate_limit)
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response

    def search_pubmed(self, query: str, max_results: int = 1000) -> List[str]:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "email": self.email,
            "api_key": self.api_key,
        }
        response = self._make_request(search_url, params)
        root = ET.fromstring(response.content)
        pmids = [pmid_elem.text for pmid_elem in root.findall(".//Id") if pmid_elem.text is not None]
        logger.info(f"Found {len(pmids)} PMIDs for query: {query}")
        return pmids

    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        pmids = [p for p in pmids if p is not None]
        if not pmids:
            return []
        batch_size = 200
        all_papers = []
        for i in range(0, len(pmids), batch_size):
            batch_pmids = [p for p in pmids[i : i + batch_size] if p is not None]
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
        try:
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            abstract_parts = [str(t) for t in (abstract_text.text for abstract_text in article.findall(".//AbstractText")) if t is not None]
            abstract = " ".join(abstract_parts)
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
        proteins = set()
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    proteins.update([str(m) for m in match if isinstance(m, str) and m is not None])
                else:
                    if isinstance(match, str) and match is not None:
                        proteins.add(str(match))
        return list(proteins)

    def extract_toxicity_terms(self, text: str) -> List[str]:
        found = []
        for term in self.toxicity_terms:
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                found.append(term)
        return found

    @abstractmethod
    def mine_proteins(self, query: str, max_results: int = 1000) -> List[ProteinToxicityResult]:
        pass

class RegexPubMedMiner(BasePubMedMiner):
    def __init__(self, email: str, api_key, rate_limit: float = 0.34, protein_dict: Optional[Set[str]] = None):
        super().__init__(email, api_key, rate_limit)
        self.protein_dict = protein_dict

    def filter_proteins(self, proteins: List[str]) -> List[str]:
        if self.protein_dict is None:
            return proteins
        return [p for p in proteins if p.upper() in self.protein_dict]

    def mine_proteins(self, query: str = None, max_results: int = 1000) -> List[ProteinToxicityResult]:
        if query is None:
            query = ""
        pmids = self.search_pubmed(query, max_results)
        papers = self.fetch_abstracts(pmids)
        results = []
        for paper in papers:
            proteins = self.extract_proteins_regex(paper["title"] + " " + paper["abstract"])
            filtered_proteins = self.filter_proteins(proteins)
            toxicity_terms = self.extract_toxicity_terms(paper["title"] + " " + paper["abstract"])
            confidence_score = 0.5 if filtered_proteins else 0.0
            result = ProteinToxicityResult(
                pmid=paper["pmid"],
                title=paper["title"],
                abstract=paper["abstract"],
                proteins=proteins,
                filtered_proteins=filtered_proteins,
                toxicity_terms=toxicity_terms,
                confidence_score=confidence_score,
                extraction_method="regex+dict",
                publication_date=paper["publication_date"],
            )
            results.append(result)
        return results

def get_uniprot_protein_names() -> Set[str]:
    UNIPROT_URL = 'https://rest.uniprot.org/uniprotkb/stream?fields=gene_names,protein_name&format=tsv&query=reviewed:true+AND+organism_id:9606'
    print('Downloading UniProt human protein names...')
    resp = requests.get(UNIPROT_URL)
    resp.raise_for_status()
    names = set()
    for line in resp.text.splitlines()[1:]:
        fields = line.split('\t')
        if len(fields) >= 2:
            gene_names = fields[0].split()
            protein_name = fields[1]
            names.update(gene_names)
            for word in protein_name.replace('-', ' ').split():
                if len(word) > 2:
                    names.add(word)
    return set(n.upper() for n in names if n)

def summarize(results: List[ProteinToxicityResult]):
    all_proteins = [p for r in results for p in r.filtered_proteins]
    unique_proteins = set(all_proteins)
    counter = Counter(all_proteins)
    print(f"Total entries: {len(results)}")
    print(f"Total protein mentions (filtered): {len(all_proteins)}")
    print(f"Unique proteins (filtered): {len(unique_proteins)}")
    print("Most common proteins:")
    for prot, count in counter.most_common(20):
        print(f"  {prot}: {count}")

def main():
    email = 'crissermel@gmail.com'  # Set your email
    api_key = '6bd3fda9334af616ddbc5b2a1b7b21a3c50' #'6bd3fda9334af616ddbc5b2a1b7b21a3c50'  # Set your NCBI API key if you have one
    # query = 'toxicity OR side effect OR adverse effect'  # Example query
    max_results = 1000
    protein_dict = get_uniprot_protein_names()
    miner = RegexPubMedMiner(email, api_key, protein_dict=protein_dict)
    tox_terms = [
        "toxicity", "adverse effect", "side effect", "hepatotoxicity", "cardiotoxicity", "nephrotoxicity",
        "neurotoxicity", "cytotoxicity", "drug-induced", "adverse drug reaction", "ADR", "drug toxicity",
        "toxic effect", "adverse event", "safety profile", "contraindication", "drug-related",
        "medication-induced", "therapeutic toxicity"
    ]
    # PMC API does not support long OR queries; search each term individually and aggregate results
    results = []
    seen_pmids = set()
    for term in tox_terms:
        query = f'"{term}"' if " " in term or "-" in term else term
        term_results = miner.mine_proteins(query, max_results=1000)
        # Deduplicate by PMID
        for r in term_results:
            if r.pmid not in seen_pmids:
                results.append(r)
                seen_pmids.add(r.pmid)
    summarize(results)
    # Save results
    with open('processed_data/protein_mining_C_results.json', 'w') as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

if __name__ == '__main__':
    main() 