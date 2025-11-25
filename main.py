import os
# GESTION DE LA MÉMOIRE GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sentence_transformers import SentenceTransformer
import psycopg 
from psycopg import Cursor
import torch

# --- CONFIGURATION ---
# 1. On pointe vers le DOSSIER et non plus un fichier unique
data_folder_path = r"Data" 

db_connection_str = "dbname=rag_chatbot user=postgres password=youssef host=localhost port=5432"
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MODEL_DIMENSION = 384 

print(f"Chargement du modèle : {model_name}...")
# Si problème de mémoire, ajoutez device='cpu'
embedding_model = SentenceTransformer(model_name)

def read_file_content(file_path: str) -> list[str]:
    """Lit un fichier texte avec gestion automatique de l'encodage (UTF-8 ou Latin-1)."""
    text = ""
    try:
        # 1. On essaie d'abord en UTF-8 (Standard moderne)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except UnicodeDecodeError:
        try:
            # 2. Si ça plante (accents), on essaie en Latin-1 (Standard Windows)
            with open(file_path, "r", encoding="latin-1") as file:
                text = file.read()
        except Exception as e:
            print(f"  [Erreur Fatale] Impossible de lire {file_path}: {e}")
            return []
    
    # Nettoyage du texte récupéré
    if text:
        text_list = text.split("\n")
        # Filtre les lignes vides et celles qui commencent par '<'
        filtered_list = [line.strip() for line in text_list if line.strip() and not line.strip().startswith("<")]
        return filtered_list
    return []

def calculate_embeddings(corpus: str, model: SentenceTransformer) -> list[float]:
    """Calcule l'embedding pour un texte donné."""
    return model.encode(corpus).tolist()

def save_embedding(corpus: str, embedding: list[float], cursor: Cursor) -> None:
    """Insère un texte et son embedding dans la base de données."""
    cursor.execute('INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)', (corpus, embedding))

def similar_corpus(input_corpus: str, model: SentenceTransformer, db_conn_str: str, limit: int = 5) -> list[tuple]:
    """Recherche les textes les plus similaires avec le FIX SQL."""
    input_embedding = calculate_embeddings(input_corpus, model)

    with psycopg.connect(db_conn_str) as conn:
        with conn.cursor() as cur:
            # --- CORRECTION MAJEURE ICI : ajout de ::vector ---
            # Cela corrige l'erreur "operator does not exist: vector <=> double precision[]"
            cur.execute("""
                SELECT id, corpus, 1 - (embedding <=> %s::vector) AS similarity
                FROM embeddings
                ORDER BY similarity DESC
                LIMIT %s
            """, (input_embedding, limit))
            results = cur.fetchall()
            return results

# --- SCRIPT D'INSERTION DE MASSE ---
def main_insertion_all_files():
    try:
        with psycopg.connect(db_connection_str) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                print("\n--- DÉBUT DE L'INITIALISATION ---")
                
                print("1. Préparation de la base de données...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("DROP TABLE IF EXISTS embeddings")
                cur.execute(f"""
                    CREATE TABLE embeddings (
                        id SERIAL PRIMARY KEY,
                        corpus TEXT,
                        embedding VECTOR({MODEL_DIMENSION})
                    )
                """)
                print("   -> Table 'embeddings' recréée à neuf.")

                # 2. Récupération de tous les fichiers .txt dans le dossier
                if not os.path.exists(data_folder_path):
                    print(f"ERREUR : Le dossier '{data_folder_path}' n'existe pas.")
                    return

                all_files = [f for f in os.listdir(data_folder_path) if f.endswith(".txt")]
                print(f"2. {len(all_files)} fichiers trouvés dans '{data_folder_path}'.")

                # 3. Boucle sur chaque fichier
                total_phrases = 0
                for index, filename in enumerate(all_files):
                    full_path = os.path.join(data_folder_path, filename)
                    print(f"   -> Traitement fichier {index+1}/{len(all_files)} : {filename}...")
                    
                    lines = read_file_content(full_path)
                    
                    if lines:
                        # Insertion des phrases de ce fichier
                        for line in lines:
                            emb = calculate_embeddings(line, embedding_model)
                            save_embedding(line, emb, cur)
                        
                        total_phrases += len(lines)
                    else:
                        print(f"      (Fichier vide ou illisible)")

                print(f"\n--- SUCCÈS TERMINAL : {total_phrases} phrases insérées au total ! ---")

    except psycopg.OperationalError as e:
        print(f"Erreur de connexion SQL : {e}")

# --- EXECUTION ---

if __name__ == "__main__":
    # 1. Lancer l'insertion de TOUS les fichiers (à faire une fois, puis commenter si besoin)
    main_insertion_all_files() 

    # 2. Test de la recherche (maintenant que la base est pleine)
    query = "Quelle est la procédure pour une demande de congé ?"
    print(f"\n--- Test de recherche pour : '{query}' ---")

    try:
        similar_results = similar_corpus(query, embedding_model, db_connection_str)
        if similar_results:
            for r_id, r_corpus, r_similarity in similar_results:
                print(f"  - ({r_similarity:.4f}) : {r_corpus}")
        else:
            print("Aucun résultat.")
    except Exception as e:
        print(f"Erreur lors de la recherche : {e}")