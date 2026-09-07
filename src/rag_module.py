"""
Modulo de RAG (Retrieval-Augmented Generation) para o assistente medico.
Implementa vector store e retriever com embeddings.
"""
import os
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SimpleDeterministicEmbeddings(Embeddings):
    """
    Embeddings deterministicos para demonstracao.
    Em producao, substitua por SentenceTransformers ou OllamaEmbeddings.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings para documentos."""
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Gera embedding para uma query."""
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        """Gera embedding deterministico baseado no conteudo."""
        # Usar hash do texto para gerar embedding deterministico
        # Isso garante que textos similares tenham embeddings similares
        text_lower = text.lower()

        # Base de palavras medicas para boost
        medical_terms = [
            'diagnostico', 'tratamento', 'sintomas', 'paciente', 'medico',
            'exame', 'rx', 'laboratorio', 'cirurgia', 'medicamento',
            'antibiotico', 'analgesico', 'hipertensao', 'diabetes',
            'cancer', 'tumor', 'infeccao', 'virus', 'bacteria'
        ]

        # Criar vetor baseado em termos medicos
        vec = np.zeros(self.dimension)

        # Adicionar features baseadas no texto
        words = text_lower.split()
        for i, word in enumerate(words[:self.dimension]):
            # Hash do palavra
            word_hash = hash(word) % self.dimension
            vec[word_hash] += 1.0

        # Boost para termos medicos
        for term in medical_terms:
            if term in text_lower:
                term_hash = hash(term) % self.dimension
                vec[term_hash] += 2.0

        # Normalizar
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()


def create_medical_documents() -> List[Document]:
    """
    Cria documentos medicos de exemplo para o vector store.

    Returns:
        Lista de Document objects
    """
    documents = [
        Document(
            page_content="""Protocolo de Pneumonia Hospitalar:
Condutas:
1. Colher culturas antes do antibiotico
2. Ceftriaxona 2g IV + Azitromicina 500mg IV
3. Reavaliacao em 48-72h
4. Ajuste conforme antibiograma
5. Oxigenoterapia para SpO2 >= 94%
6. Hidratação venosa conforme necessidade""",
            metadata={"fonte": "PROTO-001", "condicao": "Pneumonia", "especialidade": "Pneumologia"}
        ),
        Document(
            page_content="""Protocolo de Sepse:
Condutas:
1. Coletar lactato e hemoculturas
2. Antibiotico em 1 hora
3. Resuscitacao 30mL/kg cristaloides
4. Noradrenalina se PA < 90 mmHg
5. Monitorar lactato a cada 2-4h
6. De-escalation em 48-72h""",
            metadata={"fonte": "PROTO-002", "condicao": "Sepse", "especialidade": "Terapia Intensiva"}
        ),
        Document(
            page_content="""Protocolo de Insuficiencia Cardiaca:
Condutas:
1. Furosemida 40-80mg IV
2. Oxigenoterapia se SpO2 < 90%
3. Restricao hidrica 1.5-2L/dia
4. Restricao de sodio < 2g/dia
5. Monitorar balanco hidrico
6. Avaliar necessidade de vasopressores""",
            metadata={"fonte": "PROTO-003", "condicao": "Insuficiencia Cardiaca", "especialidade": "Cardiologia"}
        ),
        Document(
            page_content="""Protocolo de AVC Agudo:
Condutas:
1. TC de craneo urgente
2. Escala NIHSS
3. Trombolise ate 4.5h (tPA 0.9mg/kg)
4. Trombectomia ate 24h
5. PA < 180/105 mmHg nas primeiras 24h
6. Nao anticoagular por 24h apos tPA""",
            metadata={"fonte": "PROTO-004", "condicao": "AVC", "especialidade": "Neurologia"}
        ),
        Document(
            page_content="""Protocolo de Diabetes Mellitus Tipo 2:
Condutas:
1. HbA1c meta < 7%
2. Metformina 500-2000mg/dia
3. Dieta hipocalorica se sobrepeso
4. Exercicio 150min/semana
5. Monitoramento glicemico regular
6. Rastreamento de complicacoes anuais""",
            metadata={"fonte": "PROTO-005", "condicao": "Diabetes", "especialidade": "Endocrinologia"}
        ),
        Document(
            page_content="""Protocolo de Hipertensao Arterial:
Condutas:
1. Meta < 140/90 mmHg (geral)
2. Meta < 130/80 mmHg (diabeticos)
3. IECA ou ARA II como 1a linha
4. Tiazidicos como 2a linha
5. Restricao de sodio < 6g/dia
6. Exercicio fisico 30min/dia""",
            metadata={"fonte": "PROTO-006", "condicao": "Hipertensao", "especialidade": "Cardiologia"}
        ),
        Document(
            page_content="""Protocolo de Crise de Asma:
Condutas:
1. Salbutamol 4-8 jatos a cada 20min
2. Ipratropio brometo se moderada/grave
3. Prednisona 40-60mg VO
4. O2 umidificado para SpO2 94-98%
5. Avaliar internacao se nao melhora
6. Reavaliar em 2h""",
            metadata={"fonte": "PROTO-007", "condicao": "Asma", "especialidade": "Pneumologia"}
        ),
        Document(
            page_content="""Protocolo de Dor Lombar Aguda:
Condutas:
1. Manter atividade (evitar repouso)
2. Dipirona 1g 6/6h ou Paracetamol 750mg 6/6h
3. Ibuprofeno 600mg 8/8h
4. Fisioterapia e alongamento
5. Avaliar sinais de alerta (red flags)
6. Encaminhar se deficit neurologico""",
            metadata={"fonte": "PROTO-008", "condicao": "Dor Lombar", "especialidade": "Ortopedia"}
        )
    ]

    return documents


def create_vector_store(documents: Optional[List[Document]] = None) -> FAISS:
    """
    Cria um vector store FAISS com documentos medicos.

    Args:
        documents: Lista de documentos (opcional, usa padrao se None)

    Returns:
        Vector store FAISS
    """
    if documents is None:
        documents = create_medical_documents()

    # Configurar embeddings
    embeddings = SimpleDeterministicEmbeddings(dimension=384)

    # Dividir documentos em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(chunks)} chunks")

    # Criar vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store criado com {vectorstore.index.ntotal} vetores")

    return vectorstore


def create_retriever(vectorstore: FAISS, search_k: int = 3):
    """
    Cria um retriever a partir do vector store.

    Args:
        vectorstore: Vector store FAISS
        search_k: Numero de documentos a retornar

    Returns:
        Retriever
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k}
    )


def get_medical_context(query: str, retriever, max_length: int = 2000) -> str:
    """
    Recupera contexto medico relevante para uma query.

    Args:
        query: Pergunta do usuario
        retriever: Retriever do LangChain
        max_length: Comprimento maximo do contexto

    Returns:
        Contexto formatado
    """
    try:
        docs = retriever.invoke(query)

        context_parts = []
        current_length = 0

        for doc in docs:
            content = doc.page_content
            if current_length + len(content) <= max_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                break

        return "\n\n".join(context_parts) if context_parts else "Nenhum contexto relevante encontrado."

    except Exception as e:
        return f"Erro ao recuperar contexto: {str(e)}"
