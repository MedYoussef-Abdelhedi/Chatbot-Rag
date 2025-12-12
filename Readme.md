#  Projet RAG : Chatbot Documentaire Intelligent

Bienvenue dans ce projet de **Retrieval Augmented Generation (RAG)**. Ce système permet d'interroger une base de documents locale (fichiers texte) et d'obtenir des réponses précises générées par une Intelligence Artificielle, en évitant les hallucinations grâce à une recherche sémantique vectorielle.

##  Introduction

L'objectif de ce projet est de créer un assistant capable de :
1.  **Lire et comprendre** des documents métiers (placés dans un dossier `Data`).
2.  **Indexer** ces connaissances dans une base de données vectorielle (PostgreSQL).
3.  **Répondre** aux questions de l'utilisateur en utilisant un modèle de langage performant (LLM).

Le projet utilise **PostgreSQL (pgvector)** pour le stockage et **Qwen 2.5** comme cerveau pour la génération de texte.

---

## 📂 Description des Fichiers Source

Voici le rôle de chaque script Python présent dans ce dépôt :

### 1. `Model_embedding_plusPerformanat.py` (L'Indexeur)
Ce script est responsable de la phase de **préparation des données**.
*   **Fonction :** Il parcourt le dossier `Data/`, lit tous les fichiers `.txt`, nettoie le texte et le transforme en vecteurs numériques (embeddings).
*   **Technique :** Il utilise le modèle `paraphrase-multilingual-mpnet-base-v2` (dimension 768) pour capturer le sens des phrases.
*   **Stockage :** Il envoie ces vecteurs vers une base de données **PostgreSQL** pour permettre une recherche ultra-rapide plus tard.

### 2. `Model_LLM.py` (Le Cerveau)
Ce script gère l'**intelligence et la réponse**.
*   **Fonction :** Il charge le modèle de langage (LLM) `Qwen/Qwen2.5-1.5B-Instruct` sur la carte graphique (GPU).
*   **Logique :** Il prend la question de l'utilisateur, recherche le passage pertinent dans les documents (ou scanne les fichiers), et génère une réponse en français basée **uniquement** sur le contexte trouvé.
*   **Sécurité :** Il inclut un "Prompt Strict" pour forcer le modèle à dire "NON_TROUVE" s'il ne connaît pas la réponse, garantissant la fiabilité.

---

## 🛠️ Installation et Configuration

Suivez ces étapes pour lancer le projet sur votre machine.

### Étape 1 : Cloner le projet
```bash
git clone https://github.com/votre-compte/Chatbot-Rag.git
cd Chatbot-Rag