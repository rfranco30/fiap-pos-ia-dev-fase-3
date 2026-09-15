"""
Modulo de carregamento de modelo LLM.
Centraliza TinyLlama + PeftModel + pipeline + HuggingFacePipeline.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline

# Configuracao padrao do modelo
DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "assistente_medico_final")

# Configuracao padrao do pipeline
DEFAULT_PIPELINE_CONFIG = {
    "max_length": 2304,
    "max_new_tokens": 256,
    "truncation": True,
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """
    Carrega o tokenizer do modelo.

    Args:
        model_name: Nome do modelo no HuggingFace

    Returns:
        Tokenizer carregado
    """
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    use_finetuned: bool = True,
):
    """
    Carrega o modelo base e opcionalmente aplica o adapter LoRA fine-tuned.

    Args:
        model_name: Nome do modelo base no HuggingFace
        adapter_path: Caminho do adapter LoRA fine-tuned
        use_finetuned: Se True, aplica o adapter LoRA

    Returns:
        Modelo carregado
    """
    print("Carregando modelo base...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    if use_finetuned and adapter_path:
        print("Aplicando adapter LoRA...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("Usando modelo BASE (sem adapter)")

    model.eval()
    return model


def create_pipeline(model, tokenizer, pipeline_config: dict = None):
    """
    Cria o pipeline de geracao de texto.

    Args:
        model: Modelo carregado
        tokenizer: Tokenizer carregado
        pipeline_config: Configuracoes do pipeline (usa padrao se None)

    Returns:
        Pipeline HuggingFace
    """
    if pipeline_config is None:
        pipeline_config = DEFAULT_PIPELINE_CONFIG.copy()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        **pipeline_config,
    )
    return pipe


def load_llm(
    model_name: str = DEFAULT_MODEL_NAME,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    use_finetuned: bool = True,
    pipeline_config: dict = None,
):
    """
    Funcao principal: carrega modelo, tokenizer, pipeline e retorna LLM do LangChain.

    Args:
        model_name: Nome do modelo base
        adapter_path: Caminho do adapter LoRA
        use_finetuned: Se True, usa o modelo fine-tuned
        pipeline_config: Configuracoes extras do pipeline

    Returns:
        HuggingFacePipeline pronto para uso
    """
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, adapter_path, use_finetuned)
    pipe = create_pipeline(model, tokenizer, pipeline_config)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
