import os
import psycopg
from sentence_transformers import SentenceTransformer
from groq import Groq  # Nouvelle librairie

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Base de données (PostgreSQL)
DB_CONN_STR = "dbname=rag_chatbot user=postgres password=youssef host=localhost port=5432"

# Modèle d'Embedding Local (Reste sur votre PC pour chercher dans la DB)
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# Configuration GROQ (API)
# Note : Pour la vitesse "Instant", on utilise souvent le 8b.
# Pour l'intelligence max, on utilise le 70b.
# Ici j'ai mis le llama-3.3-70b qui est très performant, 
# mais vous pouvez mettre "llama-3.1-8b-instant" si vous voulez encore plus de vitesse.
GROQ_API_KEY = "votre_clef_api_groq_ici"  # Remplacez par votre clé API Groq
GROQ_MODEL_ID = "llama-3.3-70b-versatile" 

# ==========================================
# 2. INITIALISATION
# ==========================================
print("\n🔄 Initialisation du système RAG (Version API Groq)...")

# A. Chargement de l'Embedding (Local)
print(f"   -> Chargement du modèle d'embedding local ({EMBEDDING_MODEL_NAME})...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"   ❌ Erreur chargement Embedding : {e}")
    exit()

# B. Connexion à Groq (Distant)
print(f"   -> Connexion à l'API Groq ({GROQ_MODEL_ID})...")
try:
    client = Groq(api_key=GROQ_API_KEY)
    print("   ✅ Systèmes prêts !")
except Exception as e:
    print(f"   ❌ Erreur client Groq : {e}")
    exit()

# ==========================================
# 3. FONCTIONS
# ==========================================

def get_context_from_db(question, limit=3):
    """Transforme la question en vecteur et cherche dans PostgreSQL"""
    # Cette partie reste locale sur votre PC
    question_embedding = embedding_model.encode(question).tolist()
    
    try:
        with psycopg.connect(DB_CONN_STR) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT corpus, 1 - (embedding <=> %s::vector) AS similarity
                    FROM embeddings
                    ORDER BY similarity DESC
                    LIMIT %s
                """, (question_embedding, limit))
                results = cur.fetchall()
                return results 
    except Exception as e:
        print(f"Erreur DB : {e}")
        return []

def generate_answer_groq(question, context_text):
    """Envoie le contexte et la question à l'API Groq"""
    
    prompt = f"""
    Tu es un assistant expert et strict.
    
    CONTEXTE (Informations issues de la base de données):
    \"\"\"
    {context_text}
    \"\"\"

    QUESTION: 
    {question}
    
    CONSIGNE:
    1. Réponds à la question en te basant UNIQUEMENT sur le CONTEXTE ci-dessus.
    2. Si la réponse n'est pas dans le contexte, dis simplement : "Désolé, l'information n'est pas disponible dans les documents fournis."
    3. Réponds en Français professionnel.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant acting as a RAG system."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=GROQ_MODEL_ID,
            temperature=0.0, # 0 = Réponses factuelles, pas de créativité
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de l'appel API : {e}"

# ==========================================
# 4. BOUCLE PRINCIPALE (Mode Console)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 CHATBOT RAG x GROQ (Llama 3.3)")
    print("    Tapez 'exit' pour quitter.")
    print("="*50 + "\n")

    while True:
        # 1. Lire la question
        user_input = input("\n👉 Votre question : ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Au revoir !")
            break
            
        if not user_input.strip():
            continue

        # 2. Recherche (Retrieval - Local)
        print("   🔍 Recherche locale dans les documents...")
        found_docs = get_context_from_db(user_input, limit=13)
        
        if not found_docs:
            print("   ❌ Aucun document pertinent trouvé.")
            continue
            
        # Préparation du contexte
        context_str = "\n\n".join([f"- {doc[0]}" for doc in found_docs])
        
        # 3. Génération (Generation - Cloud Groq)
        print("   ⚡ Appel API Groq (Génération)...")
        final_response = generate_answer_groq(user_input, context_str)
        
        # 4. Affichage
        print("\n✅ RÉPONSE :")
        print("-" * 20)
        print(final_response)
        print("-" * 20)