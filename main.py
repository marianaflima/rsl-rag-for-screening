from europe_pmc_extractor import extract_from_europe_pmc
from graph_rag_assembler import assembly_graph_rag
from open_alex_extractor import extract_from_openalex
from academic_records_transformations import apply_transformations
from vector_transformer import assembly_vector_rag

QUERY_STRING = ""
FIRST_PUB_DATE = ""
LAST_PUB_DATE = ""

def main():
    extract_from_openalex(QUERY_STRING, FIRST_PUB_DATE, LAST_PUB_DATE)
    extract_from_europe_pmc(QUERY_STRING, FIRST_PUB_DATE, LAST_PUB_DATE)
    apply_transformations()
    classic_rag = assembly_vector_rag()
    graph_rag = assembly_graph_rag()

if __name__ == "__main__":
    main()
