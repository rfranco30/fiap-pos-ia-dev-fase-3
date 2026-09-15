"""
Modulo de seguranca para o assistente medico.
Implementa regras de validacao e protecao.
"""
import re
from typing import Dict, Any, Tuple


# ============================================
# REGRAS DE SEGURANCA CENTRALIZADAS
# ============================================

# Prompt de seguranca para inclusao em respostas
SAFETY_PROMPT = """
REGRAS INEGOCIAVEIS DO ASSISTENTE MEDICO:
1. NUNCA prescreva medicamentos diretamente
2. NUNCA faca diagnosticos definitivos - apenas sugira hipoteses
3. SEMPRE inclua: 'Esta resposta e uma sugestao gerada por IA e deve ser validada por um medico especialista antes da decisao clinica.'
4. SEMPRE cite a fonte do protocolo utilizado
5. Se nao tiver certeza, diga 'Nao tenho informacao suficiente para esta pergunta'
6. NUNCA compartilhe dados de pacientes identificaveis
7. SEMPRE recomende consulta com medico especialista
"""

# Palavras proibidas em respostas
PROHIBITED_WORDS = [
    'prescrever', 'receitar', 'diagnosticar definitivamente',
    'certeza absoluta', 'garantia', '100% eficaz'
]

# Palavras-chave de urgencia (consolidado de todas as fontes)
URGENCY_KEYWORDS = [
    'emergencia', 'urgente', 'parada cardiaca', 'parada respiratoria',
    'infarto', 'derrame', 'sangramento', 'choque',
    'dor toracica', 'avc', 'hemorragia', 'convulsao',
    'desmaio', 'inconsciencia', 'dificuldade para respirar',
    'reacao alergica', 'anafilaxia'
]


class MedicalSecurityValidator:
    """Validador de seguranca para respostas medicas."""

    # Palavras-chave criticas que indicam prescricao ou diagnostico definitivo
    PRESCRIPTION_KEYWORDS = [
        'prescrevo', 'prescricao', 'indico o uso de', 'tomar',
        'dose recomendada', 'posologia', 'tratamento indicado',
        'diagnostico definitivo', 'vc tem', 'voce tem',
        'esta doente com', 'confirmado que voce'
    ]

    # Termos que exigem alerta de seguranca
    SAFETY_ALERTS = [
        'urgencia', 'emergencia', 'procure imediatamente',
        'hospital', 'samu', 'upa'
    ]

    # Especialidades validas
    VALID_SPECIALTIES = [
        'clinica medica', 'cardiologia', 'pneumologia', 'neurologia',
        'ortopedia', 'endocrinologia', 'gastroenterologia', 'dermatologia',
        'psiquiatria', 'urologia', 'ginecologia', 'obstetricia',
        'pediatria', 'cirurgia', 'oftalmologia', 'otorrinolaringologia'
    ]

    @classmethod
    def validate_response(cls, response: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Valida uma resposta medica quanto a seguranca.

        Args:
            response: Texto da resposta a validar
            context: Contexto adicional (especialidade, etc.)

        Returns:
            Dict com resultado da validacao
        """
        validation_result = {
            'is_safe': True,
            'warnings': [],
            'blocked_terms': [],
            'safety_score': 100,
            'recommendations': []
        }

        response_lower = response.lower()

        # Verificar prescricao indevida
        for keyword in cls.PRESCRIPTION_KEYWORDS:
            if keyword in response_lower:
                validation_result['warnings'].append(
                    f"Termo de prescricao detectado: '{keyword}'"
                )
                validation_result['blocked_terms'].append(keyword)
                validation_result['safety_score'] -= 20

        # Verificar diagnostico definitivo
        diagnostic_patterns = [
            r'voce (tem|esta com|apresenta)',
            r'diagnostico (definitivo|confirmado)',
            r'e certeza que'
        ]
        for pattern in diagnostic_patterns:
            if re.search(pattern, response_lower):
                validation_result['warnings'].append(
                    "Possivel diagnostico definitivo detectado"
                )
                validation_result['safety_score'] -= 15

        # Verificar se tem aviso de seguranca
        safety_disclaimers = [
            'validada por um medico',
            'consulte um medico',
            'especialista',
            'aviso',
            'sugestao'
        ]
        has_disclaimer = any(d in response_lower for d in safety_disclaimers)
        if not has_disclaimer:
            validation_result['warnings'].append(
                "Resposta sem aviso de seguranca adequado"
            )
            validation_result['safety_score'] -= 10
            validation_result['recommendations'].append(
                "Adicionar: 'Esta resposta deve ser validada por um medico'"
            )

        # Verificar se cita fontes
        source_indicators = ['protocolo', 'evidencia', 'literatura', 'guideline']
        has_sources = any(s in response_lower for s in source_indicators)
        if not has_sources:
            validation_result['recommendations'].append(
                "Considerar citar fontes ou protocolos de referencia"
            )

        # Verificar termos de urgencia
        for alert in cls.SAFETY_ALERTS:
            if alert in response_lower:
                validation_result['warnings'].append(
                    f"Termo de urgencia detectado: '{alert}' - considere adicionar alerta"
                )

        # Ajustar score final
        validation_result['safety_score'] = max(0, validation_result['safety_score'])

        # Determinar se e seguro
        if validation_result['safety_score'] < 50:
            validation_result['is_safe'] = False

        return validation_result

    @classmethod
    def sanitize_response(cls, response: str) -> str:
        """
        Sanitiza uma resposta removendo termos problematicos.

        Args:
            response: Texto original

        Returns:
            Texto sanitizado
        """
        sanitized = response

        # Remover diagnostico definitivo
        patterns_to_remove = [
            r'voce (tem|esta com|apresenta) [^.]+\.?',
            r'diagnostico (definitivo|confirmado):? [^.]+\.?'
        ]
        for pattern in patterns_to_remove:
            sanitized = re.sub(pattern, '[CONSULTE UM MEDICO PARA DIAGNOSTICO]', sanitized, flags=re.IGNORECASE)

        return sanitized

    @classmethod
    def check_emergency_level(cls, symptoms: str) -> Dict[str, Any]:
        """
        Verifica nivel de urgencia baseado nos sintomas.

        Args:
            symptoms: Descricao dos sintomas

        Returns:
            Dict com nivel de urgencia e recomendacoes
        """
        symptoms_lower = symptoms.lower()

        # Sintomas de alta urgencia
        high_urgency = [
            'dor toracica', 'infarto', 'avc', 'derrame',
            'hemorragia', 'convulsao', 'desmaio', 'inconsciencia',
            'dificuldade para respirar', 'suffocacao',
            'reacao alergica', 'anafilaxia', 'choque'
        ]

        # Sintomas de urgencia media
        medium_urgency = [
            'febre alta', 'vomito persistente', 'diarreia intensa',
            'dor abdominal forte', 'sangramento', 'fratura',
            'intoxicacao', 'overdose', 'queda'
        ]

        urgency_level = 'BAIXA'
        recommendations = []

        for symptom in high_urgency:
            if symptom in symptoms_lower:
                urgency_level = 'ALTA'
                recommendations.append("PROCURE SOCORRO IMEDIATO (SAMU 192)")
                recommendations.append("Ligue para o hospital mais proximo")
                break

        if urgency_level == 'BAIXA':
            for symptom in medium_urgency:
                if symptom in symptoms_lower:
                    urgency_level = 'MEDIA'
                    recommendations.append("Procure atendimento medico em ate 24h")
                    recommendations.append("Se piorar, va a emergencia")
                    break

        if urgency_level == 'BAIXA':
            recommendations.append("Avalie com medico de confianca")
            recommendations.append("Mantenha hidratacao e repouso")

        return {
            'urgency_level': urgency_level,
            'recommendations': recommendations,
            'emergency_number': '192 (SAMU)' if urgency_level == 'ALTA' else None
        }


def create_safety_report(response: str, validation: Dict[str, Any]) -> str:
    """
    Cria relatorio de seguranca para uma resposta.

    Args:
        response: Resposta original
        validation: Resultado da validacao

    Returns:
        Relatorio formatado
    """
    report = []
    report.append("=" * 60)
    report.append("RELATORIO DE SEGURANCA")
    report.append("=" * 60)
    report.append(f"Score de Seguranca: {validation['safety_score']}/100")
    report.append(f"Status: {'SEGURO' if validation['is_safe'] else 'REQUER REVISAO'}")
    report.append("")

    if validation['warnings']:
        report.append("AVISOS:")
        for i, warning in enumerate(validation['warnings'], 1):
            report.append(f"  {i}. {warning}")
        report.append("")

    if validation['recommendations']:
        report.append("RECOMENDACOES:")
        for i, rec in enumerate(validation['recommendations'], 1):
            report.append(f"  {i}. {rec}")
        report.append("")

    report.append("=" * 60)

    return '\n'.join(report)
