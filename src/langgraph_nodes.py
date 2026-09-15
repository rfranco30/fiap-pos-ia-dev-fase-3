"""
Modulo de nos do LangGraph para o assistente medico.
Centraliza as funcoes de nodes do grafo de decisao.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


# ============================================
# ESTADO COMPARTILHADO
# ============================================
from typing import TypedDict


class MedicoState(TypedDict):
    """Estado compartilhado entre todos os nos do grafo."""
    pergunta: str
    classificacao: str
    documentos_brutos: List[Document]
    documentos_filtrados: List[Document]
    contexto: str
    resposta: str
    fontes: List[str]
    confianca: float
    etapa_atual: str
    historico_decisoes: List[str]
    alerta_necessario: bool
    mensagem_alerta: str


# ============================================
# NO 1: CLASSIFICACAO DA CONSULTA
# ============================================
def classificar_consulta(state: dict, llm) -> dict:
    """
    No de classificacao: identifica o tipo de consulta medica.
    """
    pergunta = state["pergunta"]

    prompt_classificacao = PromptTemplate(
        template="""### Instruction:
Classifique a seguinte consulta medica em UMA categoria:
- CLINICA_GERAL
- CARDIOLOGIA
- PNEUMOLOGIA
- NEUROLOGIA
- URGENCIA
- ENFERMAGEM

Consulta: {pergunta}

Responda APENAS com o nome da categoria.

### Response:
""",
        input_variables=["pergunta"]
    )

    resposta = llm.invoke(prompt_classificacao.format(pergunta=pergunta))
    classificacao = resposta.strip()

    historico = state.get("historico_decisoes", [])
    historico.append(f"Classificacao: {classificacao}")

    return {
        "classificacao": classificacao,
        "historico_decisoes": historico,
        "etapa_atual": "classificado"
    }


# ============================================
# NO 2: RECUPERACAO DE DOCUMENTOS (RAG)
# ============================================
def criar_no_recuperar(retriever):
    """
    Cria o no de recuperacao de documentos usando um retriever RAG.

    Args:
        retriever: Retriever do LangChain (vector store)

    Returns:
        Funcao do no LangGraph
    """
    def recuperar_documentos(state: dict) -> dict:
        """No de recuperacao: busca documentos relevantes na base de conhecimento."""
        pergunta = state["pergunta"]

        documentos = retriever.invoke(pergunta)

        historico = state.get("historico_decisoes", [])
        historico.append(f"Recuperados {len(documentos)} documentos relevantes")

        return {
            "documentos_brutos": documentos,
            "historico_decisoes": historico,
            "etapa_atual": "recuperado"
        }

    return recuperar_documentos


def recuperar_documentos_simulado(state: dict) -> dict:
    """No de recuperacao simulada (sem vector store)."""
    classificacao = state.get("classificacao", "CLINICA_GERAL")

    docs_simulados = [
        Document(
            page_content=f"Protocolo de {classificacao}: Condutas padronizadas para o atendimento de pacientes na area de {classificacao}.",
            metadata={"fonte": f"Protocolo_Hospitalar_{classificacao}", "versao": "2024"}
        ),
        Document(
            page_content=f"Diretrizes clinicas para {classificacao}: Baseadas em evidencias cientificas atualizadas.",
            metadata={"fonte": f"Diretrizes_{classificacao}", "versao": "2024"}
        )
    ]

    historico = state.get("historico_decisoes", [])
    historico.append(f"Recuperados {len(docs_simulados)} documentos para {classificacao}")

    return {
        "documentos_brutos": docs_simulados,
        "historico_decisoes": historico,
        "etapa_atual": "recuperado"
    }


# ============================================
# NO 3: FILTRAGEM DE RELEVANCIA
# ============================================
def filtrar_relevancia(state: dict) -> dict:
    """
    No de filtragem: valida se documentos sao relevantes para a pergunta.
    """
    documentos = state.get("documentos_brutos", [])
    pergunta = state.get("pergunta", "")

    docs_filtrados = documentos

    confianca = min(0.5 + len(docs_filtrados) * 0.15, 0.95)

    historico = state.get("historico_decisoes", [])
    historico.append(f"Filtrados {len(docs_filtrados)} documentos, confianca: {confianca:.2f}")

    return {
        "documentos_filtrados": docs_filtrados,
        "confianca": confianca,
        "historico_decisoes": historico,
        "etapa_atual": "filtrado"
    }


# ============================================
# NO 4: VALIDACAO DE QUALIDADE
# ============================================
def validar_qualidade(state: dict) -> dict:
    """
    No de validacao: verifica se a confianca e suficiente para gerar resposta.
    """
    confianca = state.get("confianca", 0)

    historico = state.get("historico_decisoes", [])

    if confianca < 0.6:
        historico.append("Validacao: confianca insuficiente, necessario reprocessar")
        return {
            "etapa_atual": "reprocessar",
            "historico_decisoes": historico
        }
    else:
        historico.append("Validacao: confianca suficiente, prosseguindo")
        return {
            "etapa_atual": "aprovado",
            "historico_decisoes": historico
        }


# ============================================
# NO 5: GERACAO DE RESPOSTA
# ============================================
def gerar_resposta(state: dict, llm) -> dict:
    """
    No de geracao: cria a resposta medica final.
    """
    pergunta = state.get("pergunta", "")
    classificacao = state.get("classificacao", "")
    documentos = state.get("documentos_filtrados", [])
    confianca = state.get("confianca", 0)

    contexto = "\n".join([doc.page_content for doc in documentos])
    fontes = [doc.metadata.get("fonte", "Desconhecida") for doc in documentos]

    prompt_resposta = PromptTemplate(
        template="""### Instruction:
Voce e um assistente medico especializado.

REGRAS:
1. Nao prescreva medicamentos diretamente
2. Nao faca diagnosticos definitivos
3. SEMPRE inclua: 'Esta resposta e uma sugestao e deve ser validada por um medico'
4. SEMPRE cite as fontes utilizadas

Contexto dos protocolos:
{contexto}

Pergunta do medico: {pergunta}
Classificacao: {classificacao}
Nivel de confianca: {confianca:.0%}

Forneça uma resposta estruturada com:
1. Resumo da analise
2. Condutas recomendadas
3. Exames complementares
4. Fontes consultadas
5. Aviso de seguranca

### Response:
""",
        input_variables=["contexto", "pergunta", "classificacao", "confianca"]
    )

    resposta = llm.invoke(prompt_resposta.format(
        contexto=contexto,
        pergunta=pergunta,
        classificacao=classificacao,
        confianca=confianca
    ))

    historico = state.get("historico_decisoes", [])
    historico.append("Resposta gerada com sucesso")

    return {
        "resposta": resposta,
        "fontes": fontes,
        "historico_decisoes": historico,
        "etapa_atual": "respondido"
    }


# ============================================
# NO 6: VERIFICACAO DE ALERTA
# ============================================
def verificar_alerta(state: dict) -> dict:
    """
    No de alerta: verifica se e necessario um alerta urgente.
    Usa check_emergency_level do security.py para avaliar urgencia.
    """
    from src.security import MedicalSecurityValidator

    pergunta = state.get("pergunta", "")

    resultado = MedicalSecurityValidator.check_emergency_level(pergunta)
    necessita_alerta = resultado["urgency_level"] in ["ALTA", "MEDIA"]

    historico = state.get("historico_decisoes", [])

    if necessita_alerta:
        mensagem_alerta = f"ALERTA {resultado['urgency_level']}: {', '.join(resultado['recommendations'])}"
        historico.append(f"Verificacao de alerta: ALERTA ATIVADO (nivel {resultado['urgency_level']})")
    else:
        mensagem_alerta = ""
        historico.append("Verificacao de alerta: Sem alerta")

    return {
        "alerta_necessario": necessita_alerta,
        "mensagem_alerta": mensagem_alerta,
        "historico_decisoes": historico,
        "etapa_atual": "finalizado"
    }


# ============================================
# FUNCOES AUXILIARES PARA NOTEBOOKS
# ============================================
def criar_grafo_basico(llm, state_class=MedicoState):
    """
    Cria um grafo LangGraph basico com todos os nos.

    Args:
        llm: LLM do LangChain
        state_class: Classe TypedDict do estado

    Returns:
        Grafo compilado
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(state_class)

    # Wrappers para injetar o LLM
    def no_classificar(state):
        return classificar_consulta(state, llm)

    def no_gerar(state):
        return gerar_resposta(state, llm)

    workflow.add_node("classificar", no_classificar)
    workflow.add_node("recuperar", recuperar_documentos_simulado)
    workflow.add_node("filtrar", filtrar_relevancia)
    workflow.add_node("validar", validar_qualidade)
    workflow.add_node("responder", no_gerar)
    workflow.add_node("alertar", verificar_alerta)

    workflow.set_entry_point("classificar")
    workflow.add_edge("classificar", "recuperar")
    workflow.add_edge("recuperar", "filtrar")
    workflow.add_edge("filtrar", "validar")

    workflow.add_conditional_edges(
        "validar",
        lambda state: state["etapa_atual"],
        {
            "aprovado": "responder",
            "reprocessar": "recuperar",
        }
    )

    workflow.add_edge("responder", "alertar")
    workflow.add_edge("alertar", END)

    return workflow.compile()


def criar_grafo_rag(llm, retriever, state_class=MedicoState):
    """
    Cria um grafo LangGraph com RAG (retrieval real).

    Args:
        llm: LLM do LangChain
        retriever: Retriever do vector store
        state_class: Classe TypedDict do estado

    Returns:
        Grafo compilado
    """
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(state_class)

    def no_classificar(state):
        return classificar_consulta(state, llm)

    def no_gerar(state):
        return gerar_resposta(state, llm)

    workflow.add_node("classificar", no_classificar)
    workflow.add_node("recuperar", criar_no_recuperar(retriever))
    workflow.add_node("filtrar", filtrar_relevancia)
    workflow.add_node("validar", validar_qualidade)
    workflow.add_node("responder", no_gerar)
    workflow.add_node("alertar", verificar_alerta)

    workflow.set_entry_point("classificar")
    workflow.add_edge("classificar", "recuperar")
    workflow.add_edge("recuperar", "filtrar")
    workflow.add_edge("filtrar", "validar")

    workflow.add_conditional_edges(
        "validar",
        lambda state: state["etapa_atual"],
        {
            "aprovado": "responder",
            "reprocessar": "recuperar",
        }
    )

    workflow.add_edge("responder", "alertar")
    workflow.add_edge("alertar", END)

    return workflow.compile()
