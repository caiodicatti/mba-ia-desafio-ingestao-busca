# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de RAG (Retrieval-Augmented Generation) que ingere um PDF, gera embeddings com Gemini e permite busca semântica via CLI.

## Pré-requisitos

- Python 3.12+
- Docker e Docker Compose
- Chave de API do Google Gemini

## Configuração

1. Copie o template de variáveis de ambiente:

```bash
cp .env.example .env
```

2. Preencha o `.env` com seus valores:

```dotenv
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=gemini_collection
PDF_PATH=document.pdf
```

3. Crie e ative o ambiente virtual:

```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

4. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

### 2. Ingerir o PDF

```bash
python src/ingest.py
```

### 3. Iniciar o chat

```bash
python src/chat.py
```

Digite sua pergunta e pressione Enter. Para encerrar, digite `sair`.

## Estrutura do Projeto

```
├── docker-compose.yml        # PostgreSQL + pgVector
├── requirements.txt          # Dependências Python
├── document.pdf              # PDF ingerido
├── src/
│   ├── gemini_embeddings.py  # Classe de embeddings via API Gemini
│   ├── ingest.py             # Ingestão do PDF no banco vetorial
│   ├── search.py             # Busca semântica + prompt RAG
│   └── chat.py               # CLI de interação
```

## Tecnologias

- **LangChain** — orquestração do pipeline RAG
- **PostgreSQL + pgVector** — banco vetorial
- **Gemini** (`gemini-embedding-001`) — geração de embeddings
- **Gemini** (`gemini-2.5-flash`) — geração de respostas
