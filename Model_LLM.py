import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
data_folder_path = r"Data" 

# On utilise le modèle 1.5 Milliards (Version 2.5, la plus performante)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Chargement du modèle {model_id} sur le GPU...")

# Chargement optimisé pour 6GB VRAM (torch_dtype="auto" va utiliser le FP16)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", 
    device_map="auto" 
)

def read_file_content(file_path: str) -> str:
    """Lit le fichier avec gestion des erreurs d'encodage (UTF-8 / Latin-1)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except:
            return ""

def ask_qwen_scan(question, context):
    """Interroge le modèle sur un morceau de texte précis."""
    
    # --- LE PROMPT STRICT ---
    # On lui donne l'instruction en Anglais (les modèles comprennent mieux les ordres techniques en anglais)
    # mais on lui demande de répondre en Français.
    prompt = f"""
    You are a strict and honest assistant.
    
    Context:
    \"\"\"
    {context}
    \"\"\"

    Instructions:
    1. Read the Context above carefully.
    2. Answer the Question below based ONLY on that Context.
    3. If the context talks about hotels, hospitals, or university enrollment but NOT about "leave request procedure" (demande de congé), you MUST reply: "NON_TROUVE".
    4. Do not invent information.
    
    Question: 
    {question}
    
    Answer (in French):
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You never hallucinate."},
        {"role": "user", "content": prompt}
    ]
    
    # Préparation de l'entrée pour le modèle
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Génération de la réponse
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=150, # On laisse un peu de place pour une réponse détaillée
        do_sample=False,    # Température 0 = Réponse factuelle et stable
        temperature=0.0,
    )
    
    # Décodage
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main_search(question):
    print(f"\n--- Démarrage du Scan avec Qwen 2.5 (1.5B) ---")
    print(f"Question : '{question}'\n")
    
    files = [f for f in os.listdir(data_folder_path) if f.endswith(".txt")]
    found_something = False
    
    # Barre de progression textuelle
    total = len(files)

    for i, filename in enumerate(files):
        path = os.path.join(data_folder_path, filename)
        content = read_file_content(path)
        
        # On ignore les fichiers trop petits (moins de 50 caractères)
        if len(content) < 50: 
            continue

        # Appel au modèle
        # print(f"Scan du fichier {filename}...", end="\r") # Décommentez pour voir le défilement
        response = ask_qwen_scan(question, content)

        # Si la réponse n'est PAS "NON_TROUVE", c'est intéressant !
        if "NON_TROUVE" not in response and len(response) > 5:
            print(f"✅ RÉSULTAT DANS '{filename}' :")
            print(f"   -> {response}\n")
            found_something = True
            
            # Optionnel : Arrêter au premier résultat trouvé
            # break 

    if not found_something:
        print("❌ Résultat : Aucune procédure trouvée dans les documents analysés.")
        print("   (Le modèle n'a pas trouvé la réponse exacte et a correctement refusé d'inventer).")

if __name__ == "__main__":
    # Votre question spécifique
    main_search("Quelle est la procédure pour une demande de congé ?")