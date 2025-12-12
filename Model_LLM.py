import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- IMPORTANT : ON IMPORTE LA RECHERCHE DEPUIS L'AUTRE FICHIER ---
# Assurez-vous que le fichier Model_embedding_plusPerformanat.py est dans le même dossier
try:
    from Model_embedding_plusPerformanat import similar_corpus, embedding_model, db_connection_str
except ImportError:
    print("ERREUR : Impossible d'importer le module de recherche. Vérifiez que 'Model_embedding_plusPerformanat.py' est bien présent.")
    exit()

# --- CONFIGURATION LLM ---
# On utilise le modèle Qwen 2.5 (1.5 Milliards)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Chargement du modèle {model_id} sur le GPU...")

# Chargement optimisé
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", 
    device_map="auto" 
)

def ask_qwen_rag(question, context):
    """
    Génère une réponse basée sur le contexte récupéré depuis la base de données.
    """
    
    # --- PROMPT GÉNÉRIQUE ---
    # J'ai rendu le prompt dynamique : il n'est plus bloqué sur "la demande de congé".
    # Il s'adapte à n'importe quelle question.
    prompt = f"""
    You are a strict and honest assistant.
    
    Context:
    \"\"\"
    {context}
    \"\"\"

    Instructions:
    1. Read the Context above carefully.
    2. Answer the Question below based ONLY on that Context.
    3. If the answer is NOT in the context, you MUST reply: "Désolé, je n'ai pas trouvé cette information dans les documents.".
    4. Answer directly in French.
    
    Question: 
    {question}
    
    Answer (in French):
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You serve accurate information."},
        {"role": "user", "content": prompt}
    ]
    
    # Préparation de l'entrée
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Génération
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=200, 
        do_sample=False,   
        temperature=0.0,
    )
    
    # Décodage
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main_rag_search(question):
    print(f"\n--- Question Utilisateur : '{question}' ---")
    
    # 1. ÉTAPE RETRIEVAL (Recherche dans PostgreSQL)
    print("1. Recherche des documents pertinents dans la base vectorielle...")
    results = similar_corpus(question, embedding_model, db_connection_str, limit=3)
    
    if not results:
        print("❌ Aucune information pertinente trouvée dans la base de données.")
        return

    # 2. PRÉPARATION DU CONTEXTE
    # On assemble les morceaux de textes trouvés (chunks) en un seul bloc
    print(f"   -> {len(results)} extraits trouvés.")
    context_text = "\n\n".join([f"Extrait {i+1}: {row[1]}" for i, row in enumerate(results)])
    
    # 3. ÉTAPE GENERATION (Appel au LLM)
    print("2. Génération de la réponse avec Qwen...")
    final_response = ask_qwen_rag(question, context_text)
    
    print("\n--- ✅ RÉPONSE FINALE ---")
    print(final_response)
    print("-------------------------")

if __name__ == "__main__":
    # Test avec la question
    main_rag_search("Quelle est la procédure pour une demande de congé ?")