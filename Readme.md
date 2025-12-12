# 🤖 Chatbot RAG : Assistant Documentaire Intelligent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PgVector-336791?style=for-the-badge&logo=postgresql)
![Groq](https://img.shields.io/badge/AI-Groq%20Llama3-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Ce projet implémente un système **RAG (Retrieval-Augmented Generation)** haute performance.  
Il permet d'interroger une base de connaissances locale (fichiers `.txt`) à l’aide d’un **chatbot intelligent**, combinant :

- 🔐 **Confidentialité** : embeddings générés localement  
- ⚡ **Performance** : génération via **Groq Llama 3.3**  
- 🧠 **Recherche sémantique** : PostgreSQL + pgvector  

---

## 🏗️ Architecture du Projet

Le système repose sur deux pipelines :

### **1️⃣ Pipeline d’Ingestion**
- Lecture des fichiers du dossier `Data/`
- Chunking et nettoyage du texte
- Embeddings via :  
  `paraphrase-multilingual-mpnet-base-v2` (768 dimensions)
- Stockage des vecteurs dans **PostgreSQL + pgvector**

### **2️⃣ Pipeline Chat (Inférence)**
- Analyse de la question utilisateur
- Similarité cosinus pour récupérer les passages pertinents
- Construction du prompt contextualisé
- Génération via **Groq Llama 3.3 (70B)**

---

## 📂 Structure du Projet

```text
Chatbot-Rag/
├── Data/                               # Base de connaissances (vos fichiers .txt)
├── main_console.py                     # Interface Console (Chat avec Groq)
├── Model_embedding_plusPerformanat.py  # Ingestion + Embedding + Indexation
├── requirements.txt                    # Dépendances Python
└── README.md                           # Documentation
🔧 Installation & Configuration1️
1️⃣ Cloner le projet
git clone https://github.com/votre-compte/Chatbot-Rag.git
cd Chatbot-Rag
2️⃣ Créer l’environnement virtuel
python -m venv venv
.\venv\Scripts\activate   # Windows
3️⃣ Installer les dépendances
pip install -r requirements.txt
🗄️ Configuration de PostgreSQL
CREATE DATABASE rag_chatbot;
2. Se connecter
\c rag_chatbot;
3. Activer pgvector
CREATE EXTENSION IF NOT EXISTS vector;
🔐 Configuration des Variables d’Environnement
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxx"

DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_chatbot
DB_USER=postgres
DB_PASSWORD=mot_de_passe

EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
🚀 Utilisation
1️⃣ Indexer vos documents
python Model_embedding_plusPerformanat.py
2️⃣ Lancer le chatbot
python main_console.py
