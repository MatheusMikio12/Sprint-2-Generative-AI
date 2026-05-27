# BANCO.md — Documentação do Banco de Dados motor.db

## Origem
Gerado por `gerar_banco.py` com seed 42. Simula 20 motores industriais com 1.500 leituras cada (total: 30.000 linhas), cobrindo operação normal e três tipos de falha com rampa progressiva de deterioração.

## Schema

### Tabela `motores`
| Coluna | Tipo | Descrição |
|---|---|---|
| id | INTEGER PK | Identificador único do motor |
| fabricante | TEXT | Fabricante (WEG, Siemens, ABB, Schneider) |
| modelo | TEXT | Modelo do motor (M100, M200, M300, X50, X100) |
| potencia_kw | REAL | Potência nominal em kW |
| data_instalacao | TEXT | Data de instalação (ISO 8601) |

### Tabela `leituras`
| Coluna | Tipo | Descrição |
|---|---|---|
| id | INTEGER PK | Identificador único da leitura |
| motor_id | INTEGER FK | Referência ao motor |
| timestamp | TEXT | Data/hora da leitura (ISO 8601, intervalo de 10 min) |
| rotacao_rpm | REAL | Rotação em RPM (nominal ~1750) |
| vibracao_mm_s | REAL | Vibração em mm/s (nominal ~2.5) |
| temperatura_c | REAL | Temperatura em °C (nominal ~65) |
| corrente_a | REAL | Corrente elétrica em A (nominal ~12) |
| falha | INTEGER | Código de falha (0-3) |

### Tabela `tipos_falha`
| id | nome | descricao |
|---|---|---|
| 0 | Normal | Operação sem falha detectada |
| 1 | Desbalanceamento | Vibração elevada por desbalanceamento do rotor; vibração sobe 5–10 mm/s sobre o nominal |
| 2 | Superaquecimento | Temperatura sobe 20–40 °C e corrente sobe 3–7 A acima do nominal |
| 3 | Falha mecânica | Deterioração de rolamentos: vibração +3–7 mm/s, temperatura +10–20 °C, rotação cai 30–80 RPM |

## Padrões de Falha (para interpretação do agente RAG)

- **Desbalanceamento (1):** vibração é o sensor principal. Valores acima de 7 mm/s com RPM instável são forte indicativo. Temperatura e corrente permanecem próximas do nominal.
- **Superaquecimento (2):** temperatura acima de 85 °C e corrente acima de 15 A são alarmes primários. Vibração permanece baixa.
- **Falha mecânica (3):** padrão misto — vibração elevada + temperatura acima do normal + queda gradual de RPM. É o tipo com sinal mais gradual (rampa mais longa).

## Desbalanceamento de Classes
- Normal (0): ~91,4% das leituras
- Desbalanceamento (1): ~2,3%
- Superaquecimento (2): ~2,6%
- Falha mecânica (3): ~3,7%

## Janelas de Deterioração
Cada falha é introduzida com uma **rampa progressiva de 30 leituras** (300 minutos) antes de atingir amplitude máxima. Isso simula a degradação real de motores e representa o maior desafio do modelo: detectar falhas no início da rampa.
