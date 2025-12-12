# 🤖 Chatbot RAG : Assistant Documentaire Intelligent

Ce projet implémente un système de **RAG (Retrieval-Augmented Generation)** haute performance. Il permet d'interagir avec une base de connaissances privée (fichiers textes) via une interface conversationnelle.

Le système combine la confidentialité des **embeddings locaux** (via SentenceTransformers) avec la puissance et la rapidité de l'API **Groq (Llama 3.3)** pour la génération de réponses.

---

## 🏗️ Architecture du Projet

Le fonctionnement repose sur deux pipelines distincts :

1.  **Pipeline d'Ingestion (Indexation)** :
    *   Lecture des documents bruts dans le dossier `Data/`.
    *   Découpage (Chunking) et nettoyage du texte.
    *   Vectorisation via le modèle local `paraphrase-multilingual-mpnet-base-v2` (Dimension 768).
    *   Stockage dans **PostgreSQL** avec l'extension `pgvector`.

2.  **Pipeline de Chat (Inférence)** :
    *   Analyse de la question utilisateur.
    *   Recherche sémantique (Cosine Similarity) dans PostgreSQL pour trouver les passages pertinents.
    *   Construction du prompt avec le contexte récupéré.
    *   Génération de la réponse via **Groq (Llama 3.3-70b)**.

---

## 📂 Structure du Projet

```text
Chatbot-Rag/
├── Data/                               # 📁 Base de connaissances (vos fichiers .txt)
├── main_console.py                     # 🚀 Interface Principale (Console + Groq API)
├── Model_embedding_plusPerformanat.py  # ⚙️ Script d'Indexation (Embedding -> DB)
├── requirements.txt                    # 📦 Dépendances Python
└── README.md                           # 📄 Documentation
🚀 Installation et Configuration
1. Cloner le projet
git clone https://github.com/votre-compte/Chatbot-Rag.git
cd Chatbot-Rag  
2. Créer l'environnement virtuel
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
3. Installer les dépendances
pip install -r requirement.txt
4. Configuration de la Base de Données (PostgreSQL)
-- 1. Créer la base de données
CREATE DATABASE rag_chatbot;

-- 2. Se connecter à la base
\c rag_chatbot

-- 3. Activer l'extension vectorielle (INDISPENSABLE)
CREATE EXTENSION IF NOT EXISTS vector;
💻 Utilisation
Étape 1 : Indexer vos documents (Ingestion)
python Model_embedding_plusPerformanat.py
Étape 2 : Lancer le Chatbot
python main_console.py
⚙️ Configuration de l'API
GROQ_API_KEY = "gsk_votre_cle_api_ici..."
📊 Performances Techniques
Embedding : sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (Dim 768).
LLM : Llama-3.3-70b-versatile via Groq (Inférence ultra-rapide).
Base de Données : PostgreSQL + pgvector (Recherche par similarité cosinus).