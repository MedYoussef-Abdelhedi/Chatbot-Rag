import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
data_folder_path = r"Data" 

# On utilise Qwen2-0.5B-Instruct
# C'est un modèle de 0.5 Milliard de paramètres (très léger, rapide)
# Il est optimisé pour suivre les instructions.
model_id = "Qwen/Qwen2-0.5B-Instruct"

print(f"Chargement du modèle {model_id} sur le GPU...")

# Chargement du Tokenizer et du Modèle
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", 
    device_map="auto" # Cela va mettre le modèle sur votre GPU automatiquement
)

def read_file_content(file_path: str) -> str:
    """Lit le fichier avec gestion des erreurs d'encodage."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except:
            return ""

def ask_qwen_with_context(question, context):
    """Demande à Qwen de répondre en utilisant UNIQUEMENT le contexte fourni."""
    
    # Prompt optimisé pour le modèle
    # On lui dit d'être concis et de dire "NON" s'il ne trouve pas.
    prompt = f"""
    Context:
    {context}

    Instruction:
    Answer the question based ONLY on the context above. 
    If the answer is not in the context, just say "NOT_FOUND".
    
    Question: 
    {question}
    
    Answer:
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You answer based on the provided text."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=100, # Réponse courte
        do_sample=False,    # Déterministe (plus précis)
        temperature=0.0,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main_search(question):
    print(f"\n--- Recherche pour : '{question}' ---")
    
    files = [f for f in os.listdir(data_folder_path) if f.endswith(".txt")]
    found_answer = False

    for i, filename in enumerate(files):
        # On lit le fichier
        path = os.path.join(data_folder_path, filename)
        content = read_file_content(path)
        
        # Si fichier trop petit, on saute
        if len(content) < 20: 
            continue

        # On interroge le modèle
        # print(f"Scan du fichier {filename}...") # Décommentez pour voir l'avancement
        response = ask_qwen_with_context(question, content)

        # Vérification de la réponse
        # Si le modèle ne répond pas "NOT_FOUND", c'est qu'il a trouvé quelque chose !
        if "NOT_FOUND" not in response and len(response) > 5:
            print(f"\n✅ TROUVÉ dans '{filename}' !")
            print(f"Réponse : {response}")
            found_answer = True
            # On peut choisir de s'arrêter au premier résultat trouvé :
            # break 
    
    if not found_answer:
        print("\n❌ Aucune réponse trouvée dans les fichiers analysés.")

if __name__ == "__main__":
    # Test
    main_search("Quelle est la procédure pour une demande de congé ?")