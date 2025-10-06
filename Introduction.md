
# Off-target (avoidome) profiling

Steps: 
- Build an off-target overview table
- Aggregate bioactivity data (human, rat, mouse)
- Explore links to off-targets (e.g., using Perplexity or another LLM-based tool)
- Start predictive modeling (QSAR, PCM)


## Overview of Off-targets (Avoidome) : “AOID-OME_TGTs”  

Create a structured table that lists the avoidome off-targets (e.g., hERG, OATP1B1, CYPs). The table should include: target, target class, species, assay type, key measurements, data source.  

Extract the list from AOID-OME_TGTs (either a provided file or internal target list). Organize by target class (e.g., Ion Channels, Enzymes, Transporters). Include orthologs (human, rat, mouse) when available.  


## Bioactivity Measurements  

Collect and compare bioactivity profiles across species to flag interspecies differences.  

Use public databases like ChEMBL, PubChem, BindingDB, AdmetSAR, and ToxCast.  
Capture compound-level data:
IC₅₀, Ki, EC₅₀, or inhibition %
Note any species-specific shifts in potency or selectivity

Entry (ex):  
Compound	Target	Human IC₅₀	Rat IC₅₀	Mouse IC₅₀	Comments

##  Find potential links to off-target effects  


Tools to use:

Perplexity.ai, PubMed, or LLM-based search tools

Search for: "Compound X off-target profile", "Compound X hERG inhibition", "OATP1B1 inhibition leads to toxicity". 

For each compound, link it to one or more off-target effects based on literature or public data.  

## Predictive modeling: QSAR, PCM  








