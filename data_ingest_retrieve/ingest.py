import sys
import chromadb
import json
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from pathlib import Path


ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text")


def add_course(course) -> None:
    document_text= f"{course['title']}({course['code']}): {course['description']}"
    metadata = {
        "code": course['code'],
        "units": course['units'],
        "prerequisite": course['prerequisite'],
        "url": course['url'],
        "grading_option": course['grading_option']
    }

    collection.add(ids = [course['code']], documents=[document_text], metadatas=[metadata])


def open_catalogs_json(file_path: str) -> list:
    page = {}
    try:
        with open(file_path, 'r') as f:
            page = json.load(f)

        return page
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_courses_json>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    
    client = chromadb.PersistentClient(path="./data/courses_db")
    collection = client.get_or_create_collection(name="courses", embedding_function=ollama_ef)

    paths = [p for p in Path(input_dir).iterdir() if p.name.endswith('index.json')]

    ids = []
    documents = []
    metadatas = []
    for path in paths:
        courses = open_catalogs_json(path)["courses"]
        for course in courses:
            add_course(course)


