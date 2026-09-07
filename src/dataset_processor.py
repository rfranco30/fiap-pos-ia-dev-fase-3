"""
Modulo de processamento de datasets para fine-tuning.
Suporta: PubMedQA, MedQuAD, e dados sinteticos.
"""
import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """Salva dados em arquivo JSON."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Salva lista de dicionarios em formato JSONL."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_pubmedqa(file_path: str, max_samples: int = 500) -> List[Dict]:
    """
    Processa o dataset PubMedQA (ori_pqal.json) para formato de treino.

    Args:
        file_path: Caminho para ori_pqal.json
        max_samples: Numero maximo de amostras a processar

    Returns:
        Lista de dicts no formato Alpaca
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # PubMedQA pode ser dict ou list
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data

    processed = []
    for idx, item in enumerate(items[:max_samples]):
        question = item.get('QUESTION', '')
        context_list = item.get('CONTEXTS', [])
        answer = item.get('LONG_ANSWER', '')
        label = item.get('reasoning_required_pred', '')

        if not question or not answer:
            continue

        # Combinar contextos
        context = ' '.join(context_list) if context_list else ''

        # Formato Alpaca
        instruction = (
            "Voce e um assistente medico especializado em pesquisa clinica. "
            "Responda a pergunta de pesquisa com base no contexto fornecido. "
            "Cite as evidencias sempre que possivel."
        )

        input_text = f"Contexto: {context[:1000]}\n\nPergunta: {question}"

        processed.append({
            'instruction': instruction,
            'input': input_text,
            'output': answer[:2000],
            'text': f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{answer[:2000]}",
            'source': 'pubmedqa',
            'id': f'pubmedqa_{idx}'
        })

    return processed


def process_medquad(medquad_dir: str, max_samples: int = 500) -> List[Dict]:
    """
    Processa o dataset MedQuAD (XML files) para formato de treino.

    Args:
        medquad_dir: Diretorio raiz do MedQuAD
        max_samples: Numero maximo de amostras a processar

    Returns:
        Lista de dicts no formato Alpaca
    """
    processed = []
    count = 0

    # Iterar sobre pastas de QA
    for folder in os.listdir(medquad_dir):
        folder_path = os.path.join(medquad_dir, folder)
        if not os.path.isdir(folder_path) or folder.startswith('.'):
            continue

        # Processar arquivos XML
        for xml_file in os.listdir(folder_path):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(folder_path, xml_file)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Extrair QA pairs - estrutura MedQuAD
                # Procurar por Question e Answer diretamente
                questions = root.findall('.//Question')
                answers = root.findall('.//Answer')

                for q, a in zip(questions, answers):
                    question = q.text.strip() if q.text else ''
                    answer = a.text.strip() if a.text else ''

                    if not question or not answer or len(answer) < 20:
                        continue

                    # Extrair categoria do nome da pasta
                    category = folder.replace('_QA', '').replace('_', ' ')

                    instruction = (
                        f"Voce e um assistente medico especializado em {category}. "
                        "Responda a pergunta de saude de forma clara, precisa e baseada em evidencias."
                    )

                    processed.append({
                        'instruction': instruction,
                        'input': question,
                        'output': answer[:2000],
                        'text': f"### Instruction:\n{instruction}\n\n### Input:\n{question}\n\n### Response:\n{answer[:2000]}",
                        'source': 'medquad',
                        'category': category,
                        'id': f'medquad_{count}'
                    })

                    count += 1
                    if count >= max_samples:
                        return processed

            except Exception as e:
                continue

    return processed


def create_synthetic_protocols() -> List[Dict]:
    """
    Cria dataset sintetico de protocolos hospitalares.

    Returns:
        Lista de dicts no formato Alpaca
    """
    protocols = [
        {
            'condition': 'Pneumonia Hospitalar',
            'instruction': 'Voce e um assistente medico especializado em Pneumologia. Responda sobre o protocolo de pneumonia hospitalar.',
            'input': 'Qual o protocolo de tratamento para pneumonia hospitalar adquirida em ambiente de UTI?',
            'output': '''Protocolo de Pneumonia Hospitalar:

1. DIAGNOSTICO:
   - Criteros clinicos: febre > 38C, tosse produtiva, infiltrado radiologico
   - Tempo de aparecimento: >= 48h apos admissao hospitalar

2. EXAMES:
   - Hemoculturas (antes do antibiotico)
   - Cultura de escarro
   - Procalcitonina
   - Gasometria arterial

3. TRATAMENTO ANTIBIOTICO:
   - Empirico: Ceftriaxona 2g IV 24/8h + Azitromicina 500mg IV 24/24h
   - Alternativa: Piperacilina-Tazobactam 4.5g IV 6/6h
   - Ajuste conforme antibiograma em 48-72h

4. MEDIDAS DE SUPORTE:
   - Oxigenoterapia para SpO2 >= 94%
   - Hidratação venosa conforme necessidade
   - Controle de dor e febre

5. REAVALIACAO:
   - Clinica em 48-72h
   - Repetir radiografia se nao melhora

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_pneumonia_1'
        },
        {
            'condition': 'Sepse',
            'instruction': 'Voce e um assistente medico especializado em Terapia Intensiva. Responda sobre o protocolo de sepse.',
            'input': 'Qual o protocolo de manejo da sepse grave e choque septico?',
            'output': '''Protocolo de Sepse e Choque Septico:

1. RECONHECIMENTO:
   - Suspeita clinica + SOFA >= 2
   - Lactato > 2 mmol/L
   - Hipotensao refrataria a fluidos

2. BUNDLE HORARIO (1a hora):
   - Coletar lactato e hemoculturas ANTES do antibiotico
   - Antibiotico de amplo espectro em 1 hora
   - Resuscitacao 30mL/kg cristaloides
   - Vasopressor se PA < 65 mmHg apos fluidos

3. ANTIBIOTICO EMPRICO:
   - Sepse comunitaria: Piperacilina-Tazobactam ou Meropenem
   - Risco de MRSA: Adicionar Vancomicina
   - Risco de fungos: Considerar Caspofungina

4. MONITORIZACAO:
   - Lactato a cada 2-4h ate normalizar
   - Hemoculturas de controle
   - Balanco hidrico rigoroso

5. TRATAMENTO DE SUCESSO:
   - De-escalation em 48-72h
   - Duração minima: 7-10 dias

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_sepse_1'
        },
        {
            'condition': 'Insuficiencia Cardiaca',
            'instruction': 'Voce e um assistente medico especializado em Cardiologia. Responda sobre o protocolo de insuficiencia cardiaca.',
            'input': 'Qual o protocolo de manejo agudo da insuficiencia cardiaca descompensada?',
            'output': '''Protocolo de Insuficiencia Cardiaca Descompensada:

1. AVALIACAO INICIAL:
   - Classificacao NYHA
   - Escala de MESSS / West Haven
   - Ecocardiograma urgente
   - BNP/NT-proBNP

2. MEDIDAS NAO FARMACOLOGICAS:
   - Posicao sentado
   - Restricao hidrica (1.5-2L/dia)
   - Restricao de sodio (< 2g/dia)
   - Balanco hidrico negativo

3. TRATAMENTO FARMACOLOGICO:
   - Furosemida 40-80mg IV (bolo inicial)
   - Nitroglicerina IV se PA > 110 mmHg
   - O2 suplementar se SpO2 < 90%
   - Acetilcisteina se edema pulmonar

4. DIAGNOSTICO DIFERENCIAL:
   - Exacerbação de DPOC
   - Pneumonia
   - Embolia pulmonar
   - Tamponamento cardiaco

5. CRITERIOS DE ALTA:
   - Clinica estavel
   - Peso seco atingido
   - Funcao renal preservada
   - Tratamento oral otimizado

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_ic_1'
        },
        {
            'condition': 'AVC Agudo',
            'instruction': 'Voce e um assistente medico especializado em Neurologia. Responda sobre o protocolo de AVC.',
            'input': 'Qual o protocolo de atendimento ao AVC isquemico agudo?',
            'output': '''Protocolo de AVC Isquemico Agudo:

1. AVALIACAO URGENTE:
   - Escala NIHSS (National Institutes of Health Stroke Scale)
   - TC de craneo sem contraste (urgente)
   - Janela terapeutica: trombolise ate 4.5h
   - Trombectomia mecanica ate 24h (se oclusao de grande vaso)

2. CRITERIOS PARA TROMBOLISE (tPA):
   - Inicio de sintomas < 4.5 horas
   - Sem hemorragia na TC
   - PA < 185/110 mmHg
   - Glicemia > 50 mg/dL

3. PROTOCOLO DE TROMBOLISE:
   - Alteplase 0.9 mg/kg (max 90mg)
   - 10% em bolo (1 min), restante em 60min
   - Monitorizacao neurologica a cada 15min

4. CUIDADOS POS-TRATAMENTO:
   - UTI neuro por 24h
   - PA < 180/105 mmHg nas primeiras 24h
   - Nao anticoagular por 24h apos tPA
   - Reabilitacao precoce

5. PREVENCAO SECUNDARIA:
   - AAS 100mg/dia ou Clopidogrel 75mg/dia
   - Estatinas
   - Controle de fatores de risco

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_avc_1'
        },
        {
            'condition': 'Diabetes Mellitus',
            'instruction': 'Voce e um assistente medico especializado em Endocrinologia. Responda sobre diabetes.',
            'input': 'Qual o protocolo de manejo do diabetes mellitus tipo 2?',
            'output': '''Protocolo de Diabetes Mellitus Tipo 2:

1. METAS TERAPEUTICAS:
   - HbA1c < 7% (individualizar)
   - Glicemia de jejum: 80-130 mg/dL
   - PPG (2h pos-prandial): < 180 mg/dL

2. MUDANCAS NO ESTILO DE VIDA:
   - Dieta hipocalorica se sobrepeso
   - Exercicio fisico 150min/semana
   - Perda de peso de 5-10% se obeso

3. TERAPIA FARMACOLOGICA:
   - 1a linha: Metformina 500-2000mg/dia
   - 2a linha (se HbA1c > 7%): SGLT2i ou GLP-1 RA
   - 3a linha: Insulinoterapia basal

4. MONITORIZACAO:
   - HbA1c a cada 3 meses
   - Automonitoramento glicemico
   - Avaliacao de complicacoes anuais

5. RASTREAMENTO DE COMPLICACOES:
   - Retinopatia: fundoscopia anual
   - Nefropatia: microalbuminuria anual
   - Neuropatia: avaliacao dos pes
   - Cardiovascular: avaliacao de risco

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_diabetes_1'
        },
        {
            'condition': 'Hipertensao Arterial',
            'instruction': 'Voce e um assistente medico especializado em Cardiologia. Responda sobre hipertensao.',
            'input': 'Qual o protocolo de tratamento da hipertensao arterial sistemica?',
            'output': '''Protocolo de Hipertensao Arterial Sistemica:

1. CLASSIFICACAO (JNC 8):
   - Normal: < 120/80 mmHg
   - Pre-hipertensao: 120-139/80-89 mmHg
   - HAS Estagio 1: 140-159/90-99 mmHg
   - HAS Estagio 2: >= 160/100 mmHg

2. META TERAPEUTICA:
   - Geral: < 140/90 mmHg
   - Diabetes: < 130/80 mmHg
   - Idosos (> 65a): < 150/90 mmHg

3. TRATAMENTO NAO FARMACOLOGICO:
   - Restricao de sodio (< 6g/dia)
   - Dieta DASH (ricas em frutas e legumes)
   - Exercicio fisico 30min/dia
   - Perda de peso se sobrepeso
   - Cessacao do tabagismo

4. FARMACOTERAPIA:
   - 1a linha: IECA ou ARA II
   - 2a linha: Tiazidicos (Hidroclorotiazida 12.5-25mg)
   - 3a linha: CCB (Anlodipino 5-10mg)
   - Combinacao: IECA + Tiazidico ou IECA + CCB

5. MONITORIZACAO:
   - PA a cada consulta
   - Funcao renal e eletritos em 2-4 semanas
   - PA ambulatorial (24h) se necessario

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_hipertensao_1'
        },
        {
            'condition': 'Asma',
            'instruction': 'Voce e um assistente medico especializado em Pneumologia. Responda sobre asma.',
            'input': 'Qual o protocolo de manejo da crise de asma aguda?',
            'output': '''Protocolo de Crise de Asma Aguda:

1. AVALIACAO DA GRAVIDADE:
   - Leve: PAE < 20% predito, SpO2 > 95%
   - Moderada: PAE 20-50%, SpO2 91-95%
   - Grave: PAE < 20%, SpO2 < 90%
   - Critica: Cianose,letargia, tira muscular

2. TRATAMENTO INICIAL:
   - Beta-2 agonista inalatorio: Salbutamol 4-8 jatos a cada 20min
   - Ipratropio brometo se crise moderada/grave
   - O2 umidificado para manter SpO2 94-98%

3. CORTICOTERAPIA:
   - Prednisona 40-60mg VO ou IV
   - Hidrocortisona 200mg IV se grave
   - Duracao: 5-7 dias

4. CRITERIOS DE MELHORA:
   - Melhora do PFE em 15-20 min
   - Reducao da frequencia respiratoria
   - Melhora da dispneia subjetiva

5. CRITERIOS DE INTERNACAO:
   - Sem melhora apos 2h de tratamento
   - PFE < 25% do predito
   - SpO2 < 90% com O2

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_asma_1'
        },
        {
            'condition': 'Dor Lombar',
            'instruction': 'Voce e um assistente medico especializado em Ortopedia. Responda sobre dor lombar.',
            'input': 'Qual o protocolo de avaliacao e tratamento da dor lombar aguda?',
            'output': '''Protocolo de Dor Lombar Aguda:

1. AVALIACAO CLINICA:
   - Historia detalhada (duracao, irradiação, fatores agravantes)
   - Exame fisico: mobilidade, pontos dolorosos, neurological
   - Sinais de alerta (red flags): perda ponderal, febre, deficit neurologico

2. EXAMES COMPLEMENTARES:
   - Radiografia de coluna: se trauma, > 50a, ou sinais de alerta
   - RM de lombar: se deficit neurologico ou radiculopatia
   - Labs: se suspeita de neoplasia ou infeccao

3. TRATAMENTO CONSERVADOR:
   - Manter atividade (evitar repouso prolongado)
   - Analgesicos: Dipirona 1g 6/6h ou Paracetamol 750mg 6/6h
   - AINEs: Ibuprofeno 600mg 8/8h (se nao contraindicado)
   - Relaxante muscular: Tizanidina 4mg 8/8h se espasmo

4. FISIOTERAPIA:
   - Alongamento e fortalecimento do core
   - Terapia manual
   - Exercicios de estabilizacao

5. INDICACOES DE ENCAMINHAMENTO:
   - Cauda equina (urgencia cirurgica)
   - Deficit neurologico progressivo
   - Dor refrataria ao tratamento

AVISO: Esta resposta e uma sugestao e deve ser validada por um medico especialista.''',
            'source': 'synthetic',
            'id': 'syn_dor_lombar_1'
        }
    ]

    return protocols


def create_synthetic_medical_qa() -> List[Dict]:
    """
    Cria dataset sintetico de perguntas e respostas medicas variadas.

    Returns:
        Lista de dicts no formato Alpaca
    """
    qa_pairs = [
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'O que e hipertensao arterial?',
            'output': 'Hipertensao arterial e uma condicao cronica caracterizada por elevacao sustentada dos niveis pressoricos sistolicos (>= 140 mmHg) e/ou diastolicos (>= 90 mmHg). E um importante fator de risco para doenças cardiovasculares, acidente vascular cerebral e doenca renal cronica. O tratamento inclui mudancas no estilo de vida e, quando necessario, farmacoterapia.',
            'source': 'synthetic',
            'id': 'syn_qa_1'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'Quais sao os sintomas de diabetes tipo 2?',
            'output': 'Os principais sintomas de diabetes tipo 2 incluem: poliuria (urinar muito), polidipsia (muita sede), polifagia (muita fome), perda de peso inexplicavel, fadiga, visao embaçada, cicatrizacao lenta de feridas, formigamento nos pes e maos. Muitos pacientes podem ser assintomaticos inicialmente, sendo diagnosticados apenas em exames de rotina.',
            'source': 'synthetic',
            'id': 'syn_qa_2'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'O que e um infarto do miocardio?',
            'output': 'Infarto do miocardio (ataque cardiaco) e a morte de tecido cardiaco devido a interrupcao do fluxo sanguineo para o coracao, geralmente por obstrucao de uma artéria coronaria por trombo. Sintomas incluem dor toracica oppressiva, irradiacao para braco esquerdo e mandibula, sudorese, nausea e dispneia. E uma emergencia medica que requer atendimento imediato.',
            'source': 'synthetic',
            'id': 'syn_qa_3'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'Como prevenir doenças cardiovasculares?',
            'output': 'A prevencao inclui: alimentacao saudavel (dieta DASH ou mediterranea), exercicio fisico regular (150min/semana), manutencao do peso saudavel (IMC 18.5-24.9), cessacao do tabagismo, controle do estresse, sono adequado (7-9h/noite), controle da pressao arterial, glicemia e colesterol, consumo moderado de alcool e vacinacao contra influenza.',
            'source': 'synthetic',
            'id': 'syn_qa_4'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'O que e pneumonia?',
            'output': 'Pneumonia e uma infeccao que inflama os alveolos pulmonares, que podem encher de pus ou liquido. Pode ser causada por bacterias, virus, fungos ou aspiracao de substancias irritantes. Sintomas incluem tosse com expectoration, febre, calafrios, dificuldade respiratoria, dor toracica ao respirar profundo e fadiga. O tratamento depende da causa e pode incluir antibioticos, antivirais ou antifungicos.',
            'source': 'synthetic',
            'id': 'syn_qa_5'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'Quais sao os tipos de diabetes?',
            'output': 'Existem principais tipos: Tipo 1 (autoimune, producao insuficiente de insulina), Tipo 2 (resistencia a insulina, mais comum), Gestacional (durante gravidez), LADA (autoimune de inicio lenta), MODY (monogenica). O Tipo 2 representa 90-95% dos casos. Todos exigem monitoramento glicemico e manejo adequado para prevenir complicacoes.',
            'source': 'synthetic',
            'id': 'syn_qa_6'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'O que e AVC (derrame cerebral)?',
            'output': 'AVC (Acidente Vascular Cerebral) e a interrupcao do fluxo sanguineo para parte do cerebro, causando morte de tecido neural. Pode ser isquico (85%, por trombo) ou hemorragico (15%, por rompimento de vaso). Sintomas: deficit motor súbito, dificuldade de fala, desvio de boca, perda de visao, forte cefaleia. E uma emergencia - tempo = cerebro.',
            'source': 'synthetic',
            'id': 'syn_qa_7'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'O que e insuficiencia cardiaca?',
            'output': 'Insuficiencia cardiaca e uma condicao cronica em que o coracao nao consegue bombear sangue suficiente para atender as necessidades metabolicas do organismo. Pode ser sistolica (FE < 40%) ou diastolica. Sintomas: dispneia de esforço e repouso, edema de membros inferiores, ortopneia, dispnotica paroxistica noturna, fadiga. Tratamento: IECA/BRA, betabloqueadores, diureticos, mudancas no estilo de vida.',
            'source': 'synthetic',
            'id': 'syn_qa_8'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'Quando devo procurar emergencia medica?',
            'output': 'Procure emergencia imediatamente se apresentar: dor toracica intensa, dificuldade respiratoria grave, deficit neurologico subito (faca torta, braco caido), convulsoes, perda de consciencia, hemorragia intensa, trauma grave, reacao alergica severa (anafilaxia), febre alta com rigidez de nuca, suicidio ou ideação suicida. Em duvida, sempre procure atendimento.',
            'source': 'synthetic',
            'id': 'syn_qa_9'
        },
        {
            'instruction': 'Voce e um assistente medico. Responda a pergunta sobre saude.',
            'input': 'O que e ansiedade?',
            'output': 'Ansiedade e uma emocao normal que se torna um transtorno quando excessiva e persistente. Sintomas: preocupacao excessiva, inquietude, irritabilidade, tensao muscular, disturbios do sono, dificuldade de concentracao, irritabilidade. Tratamento inclui psicoterapia (TCC), farmacoterapia (ISRS, IRSNA) e tecnicas de relaxamento. E importante diferenciar de outros transtornos psiquiatricos.',
            'source': 'synthetic',
            'id': 'syn_qa_10'
        }
    ]

    return qa_pairs


def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """
    Merge multiplos datasets em um so.

    Args:
        *datasets: Listas de dicionarios para merge

    Returns:
        Lista unica combinada
    """
    merged = []
    for dataset in datasets:
        merged.extend(dataset)
    return merged


def create_final_dataset(output_dir: str) -> Dict[str, int]:
    """
    Cria o dataset final de treino combinando todas as fontes.

    Args:
        output_dir: Diretorio de saida

    Returns:
        Dict com estatisticas do dataset
    """
    datasets_info = {
        'synthetic_protocols': 0,
        'synthetic_qa': 0,
        'pubmedqa': 0,
        'medquad': 0,
        'total': 0
    }

    all_data = []

    # 1. Dados sinteticos de protocolos
    protocols = create_synthetic_protocols()
    all_data.extend(protocols)
    datasets_info['synthetic_protocols'] = len(protocols)
    print(f"Protocolos sinteticos: {len(protocols)}")

    # 2. QA sintetico
    qa = create_synthetic_medical_qa()
    all_data.extend(qa)
    datasets_info['synthetic_qa'] = len(qa)
    print(f"QA sintetico: {len(qa)}")

    # 3. PubMedQA (se disponivel)
    pubmedqa_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'pubmedqa', 'ori_pqal.json')
    if os.path.exists(pubmedqa_path):
        try:
            pubmedqa_data = process_pubmedqa(pubmedqa_path, max_samples=300)
            all_data.extend(pubmedqa_data)
            datasets_info['pubmedqa'] = len(pubmedqa_data)
            print(f"PubMedQA processado: {len(pubmedqa_data)}")
        except Exception as e:
            print(f"Erro ao processar PubMedQA: {e}")

    # 4. MedQuAD (se disponivel)
    medquad_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'MedQuAD')
    if os.path.exists(medquad_dir):
        try:
            medquad_data = process_medquad(medquad_dir, max_samples=300)
            all_data.extend(medquad_data)
            datasets_info['medquad'] = len(medquad_data)
            print(f"MedQuAD processado: {len(medquad_data)}")
        except Exception as e:
            print(f"Erro ao processar MedQuAD: {e}")

    datasets_info['total'] = len(all_data)

    # Salvar dataset final
    os.makedirs(output_dir, exist_ok=True)

    # Salvar em JSONL
    jsonl_path = os.path.join(output_dir, 'dataset_fine_tuning.jsonl')
    save_jsonl(all_data, jsonl_path)

    # Salvar em formato Alpaca (JSON)
    alpaca_path = os.path.join(output_dir, 'dataset_alpaca.json')
    save_json(all_data, alpaca_path)

    # Salvar estatisticas
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    save_json(datasets_info, stats_path)

    print(f"\nDataset final criado:")
    print(f"  Total de exemplos: {datasets_info['total']}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  Alpaca: {alpaca_path}")

    return datasets_info


if __name__ == '__main__':
    # Quando executado diretamente, criar o dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    create_final_dataset(data_dir)
