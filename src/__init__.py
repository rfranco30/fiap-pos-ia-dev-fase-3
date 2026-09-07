"""
Modulo de utilitarios gerais para o projeto de Assistente Medico.
"""
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any


def anonymize_text(text: str) -> str:
    """
    Anonimiza dados sensiveis em texto medico.
    Remove: CPF, RG, CRM, nomes, datas, telefones, emails.
    """
    if not text:
        return text

    # CPF: XXX.XXX.XXX-XX
    text = re.sub(r'\d{3}\.\d{3}\.\d{3}-\d{2}', '[CPF]', text)

    # RG: XX.XXX.XXX-X
    text = re.sub(r'\d{2}\.\d{3}\.\d{3}-\d', '[RG]', text)

    # CRM: CRM/XX XXXXX
    text = re.sub(r'CRM/[A-Z]{2}\s*\d{4,6}', '[CRM]', text)

    # Telefones: (XX) XXXXX-XXXX ou XX XXXXX-XXXX
    text = re.sub(r'\(\d{2}\)\s*\d{4,5}-\d{4}', '[TELEFONE]', text)
    text = re.sub(r'\d{2}\s*\d{4,5}-\d{4}', '[TELEFONE]', text)

    # Emails
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)

    # datas no formato DD/MM/YYYY
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '[DATA]', text)

    return text


def load_json(file_path: str) -> Any:
    """Carrega arquivo JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def get_timestamp() -> str:
    """Retorna timestamp formatado."""
    return datetime.now().isoformat()
