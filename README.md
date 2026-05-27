# Sprint 2 — Manutenção Preditiva de Motores Industriais

## Estrutura do Projeto

```
sprint2/
├── gerar_banco.py              # Gera o motor.db do zero (seed=42, reproduzível)
├── motor.db                    # Banco SQLite com 30.000 leituras (20 motores x 1500)
├── BANCO.md                    # Documentação do schema e padrões de falha
├── analise_exploratoria.ipynb  # EDA completa: 8 seções, 6 gráficos, interpretações
├── treinamento.ipynb           # Pipeline ML: treino, comparação, avaliação, erros
├── modelo_falhas.joblib        # Modelo serializado (RF + metadados + métricas)
├── agente_manutencao.py        # Agente conversacional com function calling
├── requirements.txt            # Dependências Python com versões mínimas
└── README.md                   # Este arquivo
```

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

### 0. (Opcional) Recriar o banco do zero
```bash
python gerar_banco.py
```
O `motor.db` já está incluído no repositório (seed=42).

### 1. Análise Exploratória
```bash
jupyter notebook analise_exploratoria.ipynb
```

### 2. Treinamento e Avaliação
```bash
jupyter notebook treinamento.ipynb
```
O modelo é salvo automaticamente em `modelo_falhas.joblib`.

### 3. Agente Conversacional

**Demonstração (3 cenários obrigatórios, sem API key):**
```bash
python agente_manutencao.py --demo
```

**Modo interativo com LLM (requer API key):**
```bash
export ANTHROPIC_API_KEY="sua-chave-aqui"
python agente_manutencao.py
```

Exemplo de entrada:
```
Voce: motor com vibracao 8.2 mm/s, temperatura 92 C, corrente 15.5 A, rotacao 1720 RPM
```

## Variável-Alvo

| Código | Classe | Sinal Principal | Urgência |
|---|---|---|---|
| 0 | Normal | Todos dentro do nominal | Monitoramento rotineiro |
| 1 | Desbalanceamento | Vibração > 7 mm/s | Urgente |
| 2 | Superaquecimento | Temperatura > 85°C + Corrente > 15A | Crítico |
| 3 | Falha Mecânica | Vibração ↑ + Temperatura ↑ + RPM ↓ | Crítico |

## Resultados do Modelo (Random Forest)

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.99 | 1.00 | 1.00 |
| Desbalanceamento | 0.97 | 0.88 | 0.92 |
| Superaquecimento | 1.00 | 0.95 | 0.97 |
| Falha Mecânica | 0.93 | 0.92 | 0.93 |
| **F1-Macro** | | | **0.955** |

> ⚠️ Métricas avaliadas em dados **sintéticos** com padrões de separabilidade elevada.
> Gap de overfitting (treino–teste): **4,2pp** no F1-Macro — moderado e esperado.
> Em dados industriais reais, métricas tipicamente ficam entre 0.85–0.95.

## Arquitetura do Modelo

- **Algoritmo:** RandomForestClassifier (200 árvores, max_depth=20)
- **Balanceamento:** `class_weight='balanced'`
- **Features (14):** 4 sensores + rolling mean(5) + rolling std(5) + fabricante + modelo
- **Divisão:** Temporal — últimas 300 leituras/motor reservadas para teste
