from Bio import Entrez

Entrez.email = "shrechi2002@gmail.com"  # REQUIRED

def fetch_pubmed_abstracts(query, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    abstracts = []

    for pubmed_id in id_list:
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=pubmed_id,
            rettype="abstract",
            retmode="text"
        )
        abstract_text = fetch_handle.read()
        abstracts.append(abstract_text)

    return abstracts
