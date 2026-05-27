"""
gerar_banco.py
==============
Gera o banco de dados motor.db com dados simulados de sensores industriais.

Uso:
    python gerar_banco.py

Saída:
    motor.db — banco SQLite com 3 tabelas e 30.000 leituras (20 motores x 1.500)

Seed fixa (42) para reprodutibilidade total.

Estrutura dos dados simulados:
    - 20 motores com parâmetros nominais levemente distintos entre si
    - 1.500 leituras por motor em intervalos de 10 minutos
    - Falhas introduzidas com rampa progressiva de 30 leituras (~5 horas)
    - Desbalanceamento de classes: ~91% Normal, ~8.6% Falhas

Tipos de falha simulados:
    1 - Desbalanceamento : vibração sobe 5-10 mm/s acima do nominal
    2 - Superaquecimento : temperatura +20-40 C, corrente +3-7 A
    3 - Falha mecânica   : vibração +3-7 mm/s, temperatura +10-20 C, RPM cai 30-80
"""

import sqlite3
import numpy as np
import pandas as pd
import os

# ─── Configuração ────────────────────────────────────────────────────────────
SEED       = 42
N_MOTORS   = 20
N_READINGS = 1500
DB_PATH    = "motor.db"

np.random.seed(SEED)


def criar_schema(conn):
    """Cria as três tabelas do banco, apagando versões anteriores se existirem."""
    conn.executescript("""
        DROP TABLE IF EXISTS tipos_falha;
        DROP TABLE IF EXISTS leituras;
        DROP TABLE IF EXISTS motores;

        CREATE TABLE tipos_falha (
            id        INTEGER PRIMARY KEY,
            nome      TEXT NOT NULL,
            descricao TEXT
        );

        CREATE TABLE motores (
            id               INTEGER PRIMARY KEY,
            fabricante       TEXT,
            modelo           TEXT,
            potencia_kw      REAL,
            data_instalacao  TEXT
        );

        CREATE TABLE leituras (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            motor_id       INTEGER NOT NULL,
            timestamp      TEXT    NOT NULL,
            rotacao_rpm    REAL,
            vibracao_mm_s  REAL,
            temperatura_c  REAL,
            corrente_a     REAL,
            falha          INTEGER DEFAULT 0,
            FOREIGN KEY (motor_id) REFERENCES motores(id)
        );
    """)


def inserir_tipos_falha(conn):
    conn.executemany("INSERT INTO tipos_falha VALUES (?, ?, ?)", [
        (0, "Normal",
         "Operação sem falha detectada. Todos os sensores dentro dos limites nominais."),
        (1, "Desbalanceamento",
         "Desbalanceamento do rotor. Vibração sobe 5-10 mm/s sobre o nominal. "
         "RPM pode apresentar instabilidade. Temperatura e corrente permanecem próximas do nominal."),
        (2, "Superaquecimento",
         "Temperatura sobe 20-40 °C e corrente sobe 3-7 A acima do nominal. "
         "Vibração permanece baixa. Indica sobrecarga ou falha no sistema de resfriamento."),
        (3, "Falha mecânica",
         "Deterioração de rolamentos ou eixo. Vibração +3-7 mm/s, temperatura +10-20 °C, "
         "rotação cai 30-80 RPM. Padrão misto com rampa mais longa que os demais tipos."),
    ])


def inserir_motores(conn):
    """Cadastra 20 motores com parâmetros levemente variados entre si."""
    fabricantes = ["WEG", "Siemens", "ABB", "Schneider", "Weg"]
    modelos     = ["M100", "M200", "M300", "X50", "X100"]
    rows = []
    for i in range(1, N_MOTORS + 1):
        rows.append((
            i,
            np.random.choice(fabricantes),
            np.random.choice(modelos),
            round(float(np.random.uniform(5, 75)), 1),
            f"202{np.random.randint(0, 4)}-0{np.random.randint(1, 9)}"
            f"-{np.random.randint(10, 28):02d}",
        ))
    conn.executemany("INSERT INTO motores VALUES (?, ?, ?, ?, ?)", rows)
    return rows


def gerar_leituras(conn):
    """
    Gera a tabela leituras com 1.500 registros por motor.

    Lógica de simulação:
    - Parâmetros nominais levemente distintos por motor (variação de ±5%)
    - 1-2 janelas de falha por motor, sorteadas aleatoriamente entre as
      leituras 200 e 1.350 para garantir contexto antes e depois
    - Rampa progressiva: os primeiros 30 pontos de cada janela têm
      intensidade de falha crescente de 0 a 100%
    - Ruído gaussiano em todos os sensores
    """
    # Parâmetros nominais individuais por motor
    params = {
        i: {
            "rpm":  1750 + float(np.random.uniform(-50, 50)),
            "vib":   2.5 + float(np.random.uniform(-0.3, 0.3)),
            "temp":   65 + float(np.random.uniform(-5, 10)),
            "cur":    12 + float(np.random.uniform(-1, 2)),
        }
        for i in range(1, N_MOTORS + 1)
    }

    rows = []
    base_time = pd.Timestamp("2023-01-01")

    for motor_id in range(1, N_MOTORS + 1):
        p = params[motor_id]

        # Sortear 1 ou 2 janelas de falha por motor
        n_janelas = np.random.randint(1, 3)
        starts = sorted(
            np.random.choice(range(200, 1350), size=n_janelas, replace=False).tolist()
        )
        janelas = [
            (s, s + int(np.random.randint(50, 120)), int(np.random.choice([1, 2, 3])))
            for s in starts
        ]

        for t in range(N_READINGS):
            ts = (base_time + pd.Timedelta(minutes=10 * t)).isoformat()

            # Determinar tipo e intensidade da falha neste instante
            ftype, ramp = 0, 0.0
            for (s, e, tp) in janelas:
                if s <= t < e:
                    ftype = tp
                    ramp  = min(1.0, (t - s) / 30.0)   # rampa de 30 leituras
                    break

            # Leitura base com ruído
            rpm  = p["rpm"]  + np.random.normal(0, 15)
            vib  = p["vib"]  + np.random.normal(0, 0.2)
            temp = p["temp"] + np.random.normal(0, 2)
            cur  = p["cur"]  + np.random.normal(0, 0.5)

            # Desvios específicos por tipo de falha
            if ftype == 1:   # Desbalanceamento
                vib  += ramp * float(np.random.uniform(5, 10))
                rpm  += ramp * float(np.random.normal(0, 40))
            elif ftype == 2: # Superaquecimento
                temp += ramp * float(np.random.uniform(20, 40))
                cur  += ramp * float(np.random.uniform(3, 7))
            elif ftype == 3: # Falha mecânica
                vib  += ramp * float(np.random.uniform(3, 7))
                temp += ramp * float(np.random.uniform(10, 20))
                rpm  -= ramp * float(np.random.uniform(30, 80))

            rows.append((
                motor_id, ts,
                round(float(rpm),  2),
                round(float(vib),  3),
                round(float(temp), 2),
                round(float(cur),  3),
                ftype,
            ))

    conn.executemany(
        "INSERT INTO leituras "
        "(motor_id, timestamp, rotacao_rpm, vibracao_mm_s, temperatura_c, corrente_a, falha) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return rows


def imprimir_resumo(conn):
    """Exibe um resumo do banco gerado."""
    print("\n" + "=" * 50)
    print("  RESUMO DO BANCO GERADO")
    print("=" * 50)

    for tabela in ["motores", "leituras", "tipos_falha"]:
        n = pd.read_sql(f"SELECT COUNT(*) n FROM {tabela}", conn).iloc[0, 0]
        print(f"  {tabela:15s}: {n:>7,} linhas")

    print()
    dist = pd.read_sql(
        "SELECT f.nome, COUNT(*) n "
        "FROM leituras l JOIN tipos_falha f ON l.falha = f.id "
        "GROUP BY l.falha ORDER BY l.falha",
        conn,
    )
    total = dist["n"].sum()
    print("  Distribuição das classes:")
    for _, row in dist.iterrows():
        pct = row["n"] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {row['nome']:20s}: {row['n']:6,} ({pct:5.1f}%)  {bar}")

    print("=" * 50)
    print(f"  Banco salvo em: {os.path.abspath(DB_PATH)}")
    print("=" * 50 + "\n")


def main():
    print(f"Gerando {DB_PATH} com seed={SEED} ...")
    print(f"  {N_MOTORS} motores × {N_READINGS} leituras = {N_MOTORS * N_READINGS:,} registros\n")

    conn = sqlite3.connect(DB_PATH)
    try:
        criar_schema(conn)
        inserir_tipos_falha(conn)
        inserir_motores(conn)
        gerar_leituras(conn)
        conn.commit()
        imprimir_resumo(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
