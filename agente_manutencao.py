"""
agente_manutencao.py
Agente conversacional com function calling para diagnóstico de falhas em motores.
Usa a API da Anthropic com tool_use para invocar o modelo ML treinado.

Uso:
    python agente_manutencao.py

Dependências: anthropic, joblib, numpy, scikit-learn
"""

import json
import re
import joblib
import numpy as np

# ─── Carregamento do modelo ──────────────────────────────────────────────────
ARTEFATOS = joblib.load("modelo_falhas.joblib")
MODELO      = ARTEFATOS["modelo"]
FEATURE_COLS = ARTEFATOS["feature_cols"]
SENSOR_COLS  = ARTEFATOS["sensor_cols"]
LABEL_MAP    = ARTEFATOS["label_map"]   # {0:'Normal', 1:'Desbalanceamento', ...}

# Limiares nominais (médias dos dados de treinamento — aproximados)
NOMINAIS = {
    "rotacao_rpm":   1750.0,
    "vibracao_mm_s":    2.5,
    "temperatura_c":   65.0,
    "corrente_a":      12.0,
}

# ─── Definição da Tool ───────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "prever_falha_motor",
        "description": (
            "Recebe as leituras atuais de um motor industrial (rotação, vibração, "
            "temperatura e corrente) e retorna a classe de falha prevista pelo modelo "
            "de Machine Learning, junto com as probabilidades para cada classe e uma "
            "análise dos sensores fora do padrão nominal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rotacao_rpm": {
                    "type": "number",
                    "description": "Rotação do motor em RPM (valor nominal ~1750 RPM)"
                },
                "vibracao_mm_s": {
                    "type": "number",
                    "description": "Vibração em mm/s (valor nominal ~2.5 mm/s)"
                },
                "temperatura_c": {
                    "type": "number",
                    "description": "Temperatura em graus Celsius (valor nominal ~65 C)"
                },
                "corrente_a": {
                    "type": "number",
                    "description": "Corrente elétrica em Amperes (valor nominal ~12 A)"
                }
            },
            "required": ["rotacao_rpm", "vibracao_mm_s", "temperatura_c", "corrente_a"]
        }
    }
]

# ─── Implementação da Tool ───────────────────────────────────────────────────
def prever_falha_motor(rotacao_rpm: float, vibracao_mm_s: float,
                        temperatura_c: float, corrente_a: float) -> dict:
    """
    Executa a predição do modelo ML com os valores fornecidos.
    Os features de janela móvel são aproximados pelo valor pontual
    (sem histórico — contexto de uso em tempo real).
    """
    vals = {
        "rotacao_rpm":   rotacao_rpm,
        "vibracao_mm_s": vibracao_mm_s,
        "temperatura_c": temperatura_c,
        "corrente_a":    corrente_a,
    }

    # Construir vetor de features na mesma ordem do treinamento
    x = np.zeros((1, len(FEATURE_COLS)))
    for i, col in enumerate(FEATURE_COLS):
        # Identificar sensor base
        base = None
        for s in SENSOR_COLS:
            if col.startswith(s):
                base = s
                break
        if base:
            x[0, i] = vals[base]
        # fab_enc e mod_enc ficam 0 (fabricante/modelo desconhecido)

    pred_class = int(MODELO.predict(x)[0])
    probas = MODELO.predict_proba(x)[0]

    # Análise de desvio dos sensores
    desvios = {}
    alertas = []
    for sensor, valor in vals.items():
        nominal = NOMINAIS[sensor]
        desvio_pct = (valor - nominal) / nominal * 100
        desvios[sensor] = round(desvio_pct, 1)
        if abs(desvio_pct) > 15:
            direcao = "acima" if desvio_pct > 0 else "abaixo"
            alertas.append(
                f"{sensor}={valor} ({abs(desvio_pct):.0f}% {direcao} do nominal {nominal})"
            )

    return {
        "classe_prevista": pred_class,
        "diagnostico": LABEL_MAP[pred_class],
        "probabilidades": {
            LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probas)
        },
        "confianca": round(float(probas[pred_class]), 4),
        "desvios_percentuais": desvios,
        "alertas_sensores": alertas,
        "valores_entrada": vals,
        "nominais_referencia": NOMINAIS,
    }


# ─── Motor do Agente ─────────────────────────────────────────────────────────
def processar_tool_call(tool_name: str, tool_input: dict) -> str:
    """Executa a tool e retorna resultado como string JSON."""
    if tool_name == "prever_falha_motor":
        resultado = prever_falha_motor(**tool_input)
        return json.dumps(resultado, ensure_ascii=False, indent=2)
    return json.dumps({"erro": f"Tool '{tool_name}' nao reconhecida."})


def chat(historico: list, mensagem_usuario: str) -> tuple[str, list]:
    """
    Envia uma mensagem ao agente, processa tool calls se necessário,
    e retorna a resposta final + histórico atualizado.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()

        historico.append({"role": "user", "content": mensagem_usuario})

        SYSTEM = """Voce e um especialista em manutencao preditiva de motores industriais.
Quando o usuario fornecer leituras de sensores de um motor, use SEMPRE a tool
prever_falha_motor para obter o diagnostico do modelo de Machine Learning.

Apos receber o resultado da tool, explique em linguagem clara e acessivel:
1. O tipo de falha previsto e o nivel de confianca (probabilidade).
2. Quais sensores estao fora do padrao e o que isso indica fisicamente.
3. O nivel de urgencia da intervencao (critico/moderado/monitoramento).
4. Uma recomendacao pratica de acao.

Quando as probabilidades estiverem proximas entre duas classes (diferenca < 15%),
comunique explicitamente a incerteza e recomende monitoramento adicional.

Referencias dos padroes de falha:
- Desbalanceamento: vibracao >7 mm/s, RPM instavel.
- Superaquecimento: temperatura >85C, corrente >15A.
- Falha mecanica: vibracao elevada + temperatura acima do normal + queda de RPM.
"""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM,
            tools=TOOLS,
            messages=historico,
        )

        # Processar tool calls em loop
        while response.stop_reason == "tool_use":
            tool_block = next(b for b in response.content if b.type == "tool_use")
            tool_result = processar_tool_call(tool_block.name, tool_block.input)

            historico.append({"role": "assistant", "content": response.content})
            historico.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": tool_result,
                }]
            })

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=SYSTEM,
                tools=TOOLS,
                messages=historico,
            )

        resposta_final = "".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        historico.append({"role": "assistant", "content": resposta_final})
        return resposta_final, historico

    except ImportError:
        # Modo offline: usar apenas a tool local sem LLM
        return _chat_offline(historico, mensagem_usuario)


def _chat_offline(historico: list, mensagem_usuario: str) -> tuple[str, list]:
    """
    Fallback offline: extrai números da mensagem e chama a tool diretamente,
    gerando uma resposta textual sem chamar a API do LLM.
    """
    historico.append({"role": "user", "content": mensagem_usuario})
    
    # Tentar extrair valores numéricos na ordem: rotacao, vibracao, temperatura, corrente
    nums = re.findall(r"[-+]?\d*\.?\d+", mensagem_usuario)
    nums = [float(n) for n in nums]

    if len(nums) < 4:
        resp = ("Nao consegui identificar as 4 leituras. Informe: "
                "rotacao (RPM), vibracao (mm/s), temperatura (C), corrente (A).")
        historico.append({"role": "assistant", "content": resp})
        return resp, historico

    rotacao, vibracao, temperatura, corrente = nums[:4]
    resultado = prever_falha_motor(rotacao, vibracao, temperatura, corrente)

    classe = resultado["diagnostico"]
    conf   = resultado["confianca"] * 100
    probs  = resultado["probabilidades"]
    alertas = resultado["alertas_sensores"]

    URGENCIA = {
        "Normal": "MONITORAMENTO ROTINEIRO",
        "Desbalanceamento": "URGENTE — agendar correcao de balanceamento",
        "Superaquecimento": "CRITICO — verificar carga, refrigeracao e isolamento",
        "Falha mecanica": "CRITICO — inspecionar rolamentos e eixo imediatamente",
    }

    linhas = [
        f"DIAGNOSTICO: {classe} (confianca: {conf:.1f}%)",
        "",
        "Probabilidades por classe:",
    ]
    for nome, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        linhas.append(f"  {nome:<20} {prob*100:5.1f}% {bar}")

    if alertas:
        linhas += ["", "Sensores fora do padrao:"]
        for a in alertas:
            linhas.append(f"  ⚠ {a}")

    linhas += [
        "",
        f"Acao recomendada: {URGENCIA.get(classe, 'Avaliar caso')}",
    ]

    # Comunicar incerteza se houver
    sorted_probs = sorted(probs.values(), reverse=True)
    if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.15:
        linhas.append("")
        linhas.append("ATENCAO: diferenca pequena entre as duas classes mais provaveis.")
        linhas.append("Recomenda-se monitoramento continuo e coleta de mais leituras.")

    resp = "\n".join(linhas)
    historico.append({"role": "assistant", "content": resp})
    return resp, historico


# ─── Interface de Chat ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AGENTE DE MANUTENCAO PREDITIVA — Sprint 2")
    print("  Modelo: Random Forest | Base: motor.db")
    print("=" * 60)
    print()
    print("Informe as leituras do motor no formato:")
    print("  'rotacao X RPM, vibracao Y mm/s, temperatura Z C, corrente W A'")
    print("  ou apenas os 4 numeros separados por espacos/virgulas.")
    print("Digite 'sair' para encerrar.")
    print()

    historico = []
    while True:
        try:
            entrada = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando agente.")
            break

        if entrada.lower() in ("sair", "exit", "quit"):
            print("Agente encerrado.")
            break
        if not entrada:
            continue

        resposta, historico = chat(historico, entrada)
        print(f"\nAgente:\n{resposta}\n")
        print("-" * 50)


# ─── Demonstração dos 3 Cenários ─────────────────────────────────────────────
def demo_tres_cenarios():
    """
    Executa os 3 cenários obrigatórios da Sprint 2 sem interface interativa.
    """
    cenarios = [
        {
            "nome": "Cenário 1 — Motor em Operação Normal",
            "entrada": "Motor com rotacao 1748 RPM, vibracao 2.6 mm/s, temperatura 67 C, corrente 12.1 A",
            "params": (1748, 2.6, 67.0, 12.1),
        },
        {
            "nome": "Cenário 2 — Superaquecimento Claro",
            "entrada": "Motor com rotacao 1745 RPM, vibracao 2.8 mm/s, temperatura 104 C, corrente 18.5 A",
            "params": (1745, 2.8, 104.0, 18.5),
        },
        {
            "nome": "Cenário 3 — Leituras Ambíguas (incerteza)",
            "entrada": "Motor com rotacao 1720 RPM, vibracao 5.1 mm/s, temperatura 78 C, corrente 14.2 A",
            "params": (1720, 5.1, 78.0, 14.2),
        },
    ]

    print("\n" + "=" * 60)
    print("  DEMONSTRACAO — 3 CENARIOS OBRIGATORIOS (Sprint 2)")
    print("=" * 60)

    for cen in cenarios:
        print(f"\n{'─'*60}")
        print(f"  {cen['nome']}")
        print(f"{'─'*60}")
        rotacao, vibracao, temperatura, corrente = cen["params"]
        resultado = prever_falha_motor(rotacao, vibracao, temperatura, corrente)

        classe = resultado["diagnostico"]
        conf   = resultado["confianca"] * 100
        probs  = resultado["probabilidades"]
        alertas = resultado["alertas_sensores"]

        print(f"\nEntrada: {cen['entrada']}")
        print(f"\nDIAGNOSTICO: {classe} (confianca: {conf:.1f}%)")
        print("\nProbabilidades:")
        for nome, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 25)
            print(f"  {nome:<22} {prob*100:5.1f}%  {bar}")

        if alertas:
            print("\nSensores fora do padrao:")
            for a in alertas:
                print(f"  ! {a}")
        else:
            print("\nTodos os sensores dentro dos parametros nominais.")

        sorted_probs = sorted(probs.values(), reverse=True)
        if len(sorted_probs) >= 2 and (sorted_probs[0] - sorted_probs[1]) < 0.15:
            print("\n[INCERTEZA] Diferenca < 15% entre as duas classes mais provaveis.")
            print("  Recomendado: monitorar nas proximas 2h e coletar mais leituras.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo_tres_cenarios()
    else:
        demo_tres_cenarios()   # sempre mostra demo primeiro
        print()
        try:
            main()
        except Exception as e:
            print(f"Modo interativo indisponivel ({e}). Use --demo para so a demonstracao.")
