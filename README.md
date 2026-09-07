# Assistente Virtual Medico - Tech Challenge Fase 3

## Visao Geral

Sistema de assistente virtual medico construido com **LangChain**, **LangGraph** e **Fine-Tuning de LLM**, capaz de:
- Auxiliar medicos em condutas clinicas
- Responder perguntas sobre protocolos hospitalares
- Sugerir procedimentos com base em evidencias
- Automatizar fluxos de decisao com validacao de seguranca

**Tecnologias:** Python, LangChain, LangGraph, Ollama (local), FAISS, Transformers, PEFT/LoRA

**Documentacao:** [Relatorio Tecnico Completo](RELATORIO_TECNICO.md)

---

## Estrutura do Projeto

```
projeto-assistente-medico/
├── notebooks/                          # Notebooks Jupyter (execucao principal)
│   ├── 01_setup_e_preparacao_dados.ipynb
│   ├── 02_fine_tuning_llm.ipynb
│   ├── 03_langchain_fundamentos.ipynb
│   ├── 04_langgraph_rag_medico.ipynb
│   └── 05_sistema_completo.ipynb
├── src/                                # Codigo-fonte modular
│   ├── __init__.py
│   ├── dataset_processor.py            # Processamento de datasets
│   ├── security.py                     # Modulo de seguranca
│   └── rag_module.py                   # Modulo RAG
├── data/
│   ├── datasets/                       # PubMedQA, MedQuAD
│   ├── prontuarios/                    # Prontuarios anonimizados
│   └── dados_sinteticos/              # Dados sinteticos gerados
├── models/                             # Modelos fine-tuned
├── logs/                               # Logs de auditoria
├── RELATORIO_TECNICO.md                # Relatorio tecnico completo
├── requirements.txt                    # Dependencias
├── .env.example                        # Modelo de variaveis de ambiente
├── .gitignore                          # Arquivos ignorados pelo Git
└── README.md                           # Este arquivo
```

---

## Pre-requisitos

- **Python** 3.8 ou superior
- **pip** (gerenciador de pacotes)
- **Ollama** instalado (https://ollama.com/)
- Modelo baixado: `ollama pull llama3.2`
- (Opcional) **GPU NVIDIA** para fine-tuning (recomendado: min. 8GB VRAM)

---

## Instalacao

### 1. Instalar Ollama

O Ollama permite rodar modelos de linguagem localmente, sem precisar de chaves de API.

**macOS:**
```bash
# Opcao 1: Via terminal (recomendado)
curl -fsSL https://ollama.com/install.sh | sh

# Opcao 2: Baixar o instalador em https://ollama.com/download
# Escolha a versao para macOS (Apple Silicon ou Intel)
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
- Baixe o instalador em https://ollama.com/download
- Execute o arquivo `.exe` e siga as instrucoes

**Apos a instalacao, verifique se esta funcionando:**
```bash
ollama --version
```

### 2. Baixar o Modelo

O projeto utiliza o modelo **llama3.2** (recomendado) ou alternativas menores:

```bash
# Modelo principal (recomendado - ~2GB)
ollama pull llama3.2

# Alternativas menores (se tiver pouca memoria RAM):
# ollama pull llama3.2:1b    # ~1GB
# ollama pull phi3           # ~2GB
# ollama pull mistral        # ~4GB

# Verificar modelos instalados
ollama list
```

### 3. Iniciar o Servico Ollama

```bash
# Iniciar o servidor (deixe rodando em um terminal separado)
ollama serve

# O servidor ficara disponivel em: http://localhost:11434
```

**Dica:** Em macOS, o Ollama ja inicia automaticamente apos a instalacao. Verifique se o icone aparece na barra de tarefas.

### 4. Testar o Ollama

```bash
# Testar via terminal
ollama run llama3.2 "Ola, tudo bem?"

# Ou via API (em outro terminal)
curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"Ola"}'
```

### 5. Configurar o Projeto

```bash
# Navegar ate o diretorio do projeto
cd /Users/rodrigofranco/Environment/FIAP/fase\ 3/aulas/projeto-assistente-medico/

# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variaveis de ambiente
cp .env.example .env

# Criar diretorio de logs
mkdir -p logs
```

### 6. Verificar a Configuracao

```bash
# Verificar se o Ollama esta rodando
curl http://localhost:11434/api/tags

# Deve retornar a lista de modelos disponiveis
```

---

## Execucao dos Notebooks

### Iniciar o Jupyter

```bash
jupyter notebook notebooks/
```

Ou, se preferir o JupyterLab:

```bash
jupyter lab notebooks/
```

### Ordem de execucao

Execute os notebooks **na ordem numerica** (01 a 05):

| # | Notebook | Descricao | Tempo estimado |
|---|----------|-----------|----------------|
| 01 | `01_setup_e_preparacao_dados.ipynb` | Download PubMedQA/MedQuAD, parse, anonimizacao, JSONL | ~5-10 min |
| 02 | `02_fine_tuning_llm.ipynb` | Fine-tuning com QLoRA/LoRA | ~30-60 min* |
| 03 | `03_langchain_fundamentos.ipynb` | Prompts, Chains, Loaders, Agents | ~10 min |
| 04 | `04_langgraph_rag_medico.ipynb` | LangGraph, RAG com FAISS | ~10 min |
| 05 | `05_sistema_completo.ipynb` | Sistema integrado com seguranca | ~10 min |

*\*O fine-tuning (notebook 02) pode demorar dependendo do hardware. Em GPU, ~30min. Em CPU, pode levar horas.*

---

## Conteudo dos Notebooks

### Notebook 01 - Setup e Preparacao dos Dados
- Configuracao do ambiente
- **Download do PubMedQA** (github.com/pubmedqa/pubmedqa) - 1,000 QA pairs de pesquisa clinica
- **Download do MedQuAD** (github.com/abachaa/MedQuAD) - 47,457 QA pairs do NIH
- Parse de XML para JSON
- Conversao para formato fine-tuning (JSONL e Alpaca)
- Funcoes de anonimizacao (LGPD)
- Analise de qualidade do dataset

### Notebook 02 - Fine-Tuning de LLM
- Configuracao do modelo base (TinyLlama)
- Quantizacao 4-bit (QLoRA) para eficiencia de memoria
- Configuracao de LoRA (rank 16, alpha 32)
- Treinamento com SFTTrainer
- Avaliacao (loss, perplexidade)
- Salvamento do modelo fine-tuned

### Notebook 03 - LangChain Fundamentos
- Prompt Engineering com PromptTemplate
- LLMChain para analise clinica
- SequentialChain para pipeline completo
- Document Loaders (TextLoader, DirectoryLoader)
- Text Splitters para chunking
- Agents com ferramentas (ReAct)
- Output Parsers para estruturacao

### Notebook 04 - LangGraph e RAG
- Definicao de estado com TypedDict
- Implementacao de nos: classificar, recuperar, filtrar, validar, responder, alertar
- Arestas condicionais (fluxo baseado em confianca)
- Vector Store com FAISS para busca semantica
- Integracao RAG (Retrieval-Augmented Generation)
- Padrao ReAct (Reasoning + Acting)

### Notebook 05 - Sistema Completo
- Integracao de todos os componentes
- Sistema de logging para auditoria
- Modulo de seguranca (regras medicas)
- Anonimizacao de dados sensiveis
- Validacao de respostas
- Detectao de urgencia
- Explainability (explicabilidade)
- Demonstracao com 3 cenarios de teste

---

## Seguranca e Conformidade

### Regras Implementadas
1. **NUNCA** prescreve medicamentos diretamente
2. **NUNCA** faz diagnosticos definitivos
3. **SEMPRE** inclui aviso de validacao humana
4. **SEMPRE** cita fontes dos protocolos
5. Recomenda consulta com especialista

### Protecao de Dados
- Anonimizacao de CPF, RG, CRMs
- Mascaramento de nomes e datas
- Logs sem dados identificaveis
- Conformidade com LGPD

### Auditoria
- Todas as consultas sao registradas em `logs/auditoria.jsonl`
- Historico completo de decisoes
- Pontuacao de seguranca por resposta
- Timestamp de cada interacao

---

## Dependencias Principais

| Biblioteca | Versao | Funcao |
|------------|--------|--------|
| `langgraph` | >=0.2.0 | Grafos de decisao |
| `langchain` | >=0.3.0 | Framework para LLMs |
| `langchain-ollama` | >=0.2.0 | Integracao Ollama |
| `langchain-community` | >=0.3.0 | Comunidade LangChain |
| `faiss-cpu` | >=1.8.0 | Vector store |
| `transformers` | >=4.40.0 | Modelos de linguagem |
| `peft` | >=0.11.0 | Fine-tuning eficiente |
| `bitsandbytes` | >=0.43.0 | Quantizacao |
| `datasets` | >=2.19.0 | Datasets HuggingFace |
| `torch` | >=2.0.0 | Deep learning |

---

## Solucao de Problemas

### Erro: `Connection refused` ou `Ollama not running`
```bash
# Verificar se o Ollama esta rodando
curl http://localhost:11434/api/tags

# Se nao estiver, inicie o servidor
ollama serve
```

### Erro: `Model not found`
```bash
# Verificar modelos instalados
ollama list

# Baixar o modelo necessario
ollama pull llama3.2

# Ou listar modelos disponiveis para download
ollama list | grep llama
```

### Erro: `Timeout` ou resposta muito lenta
- Ollama roda localmente, o tempo depende do hardware
- Primeira execucao e mais lenta (carrega o modelo na RAM)
- Use modelos menores se tiver menos de 8GB de RAM

### Erro: `ModuleNotFoundError`
```bash
# Verificar se o ambiente virtual esta ativado
source venv/bin/activate

# Reinstalar dependencias
pip install -r requirements.txt
```

### Erro de memoria no fine-tuning
- Use Google Colab com GPU (T4 ou superior)
- Reduza `per_device_train_batch_size` no notebook 02
- O modelo TinyLlama (~1.1B param) e leve; para modelos maiores, use QLoRA

### Jupyter nao inicia
```bash
pip install jupyter
jupyter notebook notebooks/
```
Acesse http://localhost:8888 no navegador

---

## Referencias

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://ollama.com/)
- [FAISS](https://faiss.ai/)
- [HuggingFace PEFT](https://huggingface.co/docs/peft/)

---

**Autor:** Rodrigo Franco Santana  
**Matricula:** RM372486  
**Projeto:** Tech Challenge Fase 3 - FIAP  
**Data de Entrega:** 14/09/2026
