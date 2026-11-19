import sqlite3
import json
import csv
import hashlib
import unicodedata
import bcrypt
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict


class BancoDados:
    MAX_TENTATIVAS_LOGIN = 3
    MINUTOS_BLOQUEIO_LOGIN = 2

    def __init__(self, db_path: str = "sistema_bancario_unip.db"):
        self.db_path = db_path
        self._inicializar_tabelas()
        self._criar_contas_grupo_e_cofre()

    def _conectar(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _gerar_checksum_vetor(embedding_json: str) -> str:
        """Gera assinatura SHA-256 para garantir integridade do vetor 128D no banco."""
        return hashlib.sha256(embedding_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalizar_texto(texto: str) -> str:
        nfkd = unicodedata.normalize("NFKD", texto.strip().lower())
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _inicializar_tabelas(self):
        with self._conectar() as conn:
            cursor = conn.cursor()

            # Tabela de clientes: senhas em bcrypt (RF-02) e vetor 128D sem fotos (RNF-03)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT NOT NULL,
                    numero_conta TEXT UNIQUE NOT NULL,
                    login_alias TEXT UNIQUE,
                    senha_hash BLOB NOT NULL,
                    embedding_128d TEXT NOT NULL,
                    embedding_sha256 TEXT NOT NULL,
                    biometria_cadastrada INTEGER NOT NULL DEFAULT 0,
                    saldo REAL NOT NULL DEFAULT 25000.00,
                    tentativas_falhas INTEGER NOT NULL DEFAULT 0,
                    bloqueado_ate TEXT,
                    criado_em TEXT NOT NULL
                )
            """)

            # Migração automática caso já exista um .db anterior na pasta
            cursor.execute("PRAGMA table_info(usuarios)")
            colunas_existentes = {row["name"] for row in cursor.fetchall()}
            if "login_alias" not in colunas_existentes:
                cursor.execute("ALTER TABLE usuarios ADD COLUMN login_alias TEXT")
            if "biometria_cadastrada" not in colunas_existentes:
                cursor.execute("ALTER TABLE usuarios ADD COLUMN biometria_cadastrada INTEGER NOT NULL DEFAULT 1")

            # Tabela de transações e auditoria biométrica completa
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transacoes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conta_origem TEXT NOT NULL,
                    conta_destino TEXT NOT NULL,
                    valor REAL NOT NULL,
                    nivel_seguranca TEXT NOT NULL,
                    exigiu_biometria INTEGER NOT NULL,
                    liveness_aprovado INTEGER NOT NULL DEFAULT 0,
                    distancia_euclidiana REAL,
                    similaridade_cosseno REAL,
                    latencia_ms REAL,
                    status TEXT NOT NULL,
                    data_hora TEXT NOT NULL
                )
            """)
            conn.commit()

    def _criar_contas_grupo_e_cofre(self):
        """
        Cria automaticamente os 5 integrantes do grupo (ETP) com senha '1234'
        e as contas destinatárias (Conta Padrão e Cofre Central Nível 3).
        """
        integrantes_grupo = [
            ("Gustavo Henrique Sousa da Silva", "1001-0", "gustavo", "1234", 25000.00),
            ("Higor Fernandes dos Santos Moreira", "1002-0", "higor", "1234", 25000.00),
            ("LUCAS SILVA SABINO", "1003-0", "lucas", "1234", 25000.00),
            ("Moisés Paixão Pereira", "1004-0", "moises", "1234", 25000.00),
            ("Guilherme Silva Tavares", "1005-0", "guilherme", "1234", 25000.00),
        ]

        contas_sistema = [
            ("Conta Destino Padrão (UNIP)", "9999-0", "destino", "1234", 50000.00),
            ("Cofre Central - Ministério (Nível 3)", "8888-0", "cofre", "1234", 100000.00),
        ]

        with self._conectar() as conn:
            cursor = conn.cursor()
            for nome, conta, alias, senha_plana, saldo in integrantes_grupo + contas_sistema:
                cursor.execute("SELECT id FROM usuarios WHERE numero_conta = ?", (conta,))
                if not cursor.fetchone():
                    senha_hash = bcrypt.hashpw(senha_plana.encode("utf-8"), bcrypt.gensalt(rounds=12))
                    vetor_json = json.dumps([0.0] * 128)
                    checksum = self._gerar_checksum_vetor(vetor_json)
                    cursor.execute("""
                        INSERT INTO usuarios (
                            nome, numero_conta, login_alias, senha_hash, embedding_128d,
                            embedding_sha256, biometria_cadastrada, saldo, criado_em
                        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """, (
                        nome, conta, alias, senha_hash, vetor_json, checksum, saldo,
                        datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    ))
            conn.commit()

    def cadastrar_usuario(
        self,
        nome: str,
        numero_conta: str,
        senha_plana: str,
        embedding: np.ndarray,
        saldo_inicial: float = 25000.00
    ) -> Tuple[bool, str]:
        """Cadastra cliente com hash bcrypt + salt (RF-02) e vetor 128D (RNF-03)."""
        if embedding.shape != (128,):
            return False, "Erro: O vetor biométrico deve possuir exatamente 128 dimensões."
        if len(senha_plana) < 4:
            return False, "A senha deve ter no mínimo 4 caracteres."

        try:
            salt = bcrypt.gensalt(rounds=12)
            senha_hash = bcrypt.hashpw(senha_plana.encode("utf-8"), salt)

            embedding_json = json.dumps(embedding.astype(float).tolist())
            checksum = self._gerar_checksum_vetor(embedding_json)
            alias = f"conta_{numero_conta.strip()}"

            with self._conectar() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO usuarios (
                        nome, numero_conta, login_alias, senha_hash, embedding_128d,
                        embedding_sha256, biometria_cadastrada, saldo, criado_em
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """, (
                    nome.strip(),
                    numero_conta.strip(),
                    alias,
                    senha_hash,
                    embedding_json,
                    checksum,
                    saldo_inicial,
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                ))
                conn.commit()

            return True, "Conta digital e template biométrico 128D cadastrados com sucesso!"
        except sqlite3.IntegrityError:
            return False, "Número de conta já cadastrado no sistema."
        except Exception as e:
            return False, f"Erro interno ao cadastrar cliente: {str(e)}"

    def atualizar_biometria_usuario(self, numero_conta: str, embedding: np.ndarray) -> Tuple[bool, str]:
        """Grava ou atualiza o vetor 128D (RNF-03) de um integrante já cadastrado."""
        if embedding.shape != (128,):
            return False, "Erro: O vetor biométrico deve conter exatamente 128 dimensões."

        try:
            embedding_json = json.dumps(embedding.astype(float).tolist())
            checksum = self._gerar_checksum_vetor(embedding_json)

            with self._conectar() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE usuarios
                    SET embedding_128d = ?, embedding_sha256 = ?, biometria_cadastrada = 1
                    WHERE numero_conta = ?
                """, (embedding_json, checksum, numero_conta))
                conn.commit()

            return True, "Biometria facial (vetor 128D) vinculada com sucesso à sua conta!"
        except Exception as e:
            return False, f"Erro ao salvar vetor biométrico: {str(e)}"

    def autenticar_login(self, identificador: str, senha_plana: str) -> Tuple[bool, Optional[Dict], str]:
        """
        Valida credenciais aceitando tanto o número da conta (ex: '1003-0')
        quanto o primeiro nome do integrante (ex: 'Lucas', 'Higor', 'Igor').
        """
        chave = self._normalizar_texto(identificador)
        if chave == "igor":
            chave = "higor"

        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM usuarios
                WHERE LOWER(numero_conta) = ? OR LOWER(login_alias) = ?
            """, (chave, chave))
            usuario = cursor.fetchone()

            if not usuario:
                return False, None, "Conta ou usuário não encontrado."

            if usuario["bloqueado_ate"]:
                limite = datetime.strptime(usuario["bloqueado_ate"], "%Y-%m-%d %H:%M:%S")
                if datetime.now() < limite:
                    restante = int((limite - datetime.now()).total_seconds())
                    return False, None, f"Conta temporariamente bloqueada por segurança. Aguarde {restante}s."
                else:
                    cursor.execute(
                        "UPDATE usuarios SET tentativas_falhas = 0, bloqueado_ate = NULL WHERE id = ?",
                        (usuario["id"],)
                    )
                    conn.commit()

            if self._gerar_checksum_vetor(usuario["embedding_128d"]) != usuario["embedding_sha256"]:
                return False, None, "Alerta de Integridade: Template biométrico adulterado no banco."

            if bcrypt.checkpw(senha_plana.encode("utf-8"), usuario["senha_hash"]):
                cursor.execute(
                    "UPDATE usuarios SET tentativas_falhas = 0, bloqueado_ate = NULL WHERE id = ?",
                    (usuario["id"],)
                )
                conn.commit()
                dados = {
                    "id": usuario["id"],
                    "nome": usuario["nome"],
                    "numero_conta": usuario["numero_conta"],
                    "saldo": usuario["saldo"],
                    "biometria_cadastrada": bool(usuario["biometria_cadastrada"]),
                    "embedding": np.array(json.loads(usuario["embedding_128d"]), dtype=np.float64)
                }
                return True, dados, "Autenticação primária autorizada."

            novas_falhas = usuario["tentativas_falhas"] + 1
            if novas_falhas >= self.MAX_TENTATIVAS_LOGIN:
                bloqueio = (datetime.now() + timedelta(minutes=self.MINUTOS_BLOQUEIO_LOGIN)).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "UPDATE usuarios SET tentativas_falhas = ?, bloqueado_ate = ? WHERE id = ?",
                    (novas_falhas, bloqueio, usuario["id"])
                )
                conn.commit()
                return False, None, f"Senha incorreta. Conta bloqueada por {self.MINUTOS_BLOQUEIO_LOGIN} min."

            cursor.execute(
                "UPDATE usuarios SET tentativas_falhas = ? WHERE id = ?",
                (novas_falhas, usuario["id"])
            )
            conn.commit()
            restantes = self.MAX_TENTATIVAS_LOGIN - novas_falhas
            return False, None, f"Senha incorreta. Restam {restantes} tentativa(s)."

    def verificar_senha_adicional(self, numero_conta: str, senha_plana: str) -> bool:
        """Revalidação de senha exigida para operações críticas do Cofre (Nível 3)."""
        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT senha_hash FROM usuarios WHERE numero_conta = ?", (numero_conta,))
            row = cursor.fetchone()
            if not row:
                return False
            return bcrypt.checkpw(senha_plana.encode("utf-8"), row["senha_hash"])

    def obter_dados_atualizados(self, numero_conta: str) -> Optional[Dict]:
        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM usuarios WHERE numero_conta = ?", (numero_conta,))
            usuario = cursor.fetchone()

        if not usuario:
            return None

        return {
            "id": usuario["id"],
            "nome": usuario["nome"],
            "numero_conta": usuario["numero_conta"],
            "saldo": usuario["saldo"],
            "biometria_cadastrada": bool(usuario["biometria_cadastrada"]),
            "embedding": np.array(json.loads(usuario["embedding_128d"]), dtype=np.float64)
        }

    def validar_destinatario(self, conta_destino: str) -> Tuple[bool, str]:
        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM usuarios WHERE numero_conta = ?", (conta_destino.strip(),))
            row = cursor.fetchone()
            if row:
                return True, row["nome"]
            return False, ""

    def executar_transferencia(
        self,
        conta_origem: str,
        conta_destino: str,
        valor: float,
        nivel_seguranca: str,
        exigiu_biometria: bool,
        liveness_aprovado: bool = False,
        distancia_euclidiana: Optional[float] = None,
        similaridade_cosseno: Optional[float] = None,
        latencia_ms: Optional[float] = None
    ) -> Tuple[bool, str]:
        """RF-01: Transação ACID garantindo consistência entre débito e crédito."""
        if valor <= 0:
            return False, "O valor da transferência deve ser maior que zero."
        if conta_origem == conta_destino:
            return False, "A conta de destino não pode ser igual à conta de origem."

        conn = self._conectar()
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")

            cursor.execute("SELECT saldo FROM usuarios WHERE numero_conta = ?", (conta_origem,))
            origem = cursor.fetchone()
            if not origem or origem["saldo"] < valor:
                conn.rollback()
                return False, "Saldo insuficiente para realizar a operação."

            cursor.execute("SELECT saldo FROM usuarios WHERE numero_conta = ?", (conta_destino,))
            destino = cursor.fetchone()
            if not destino:
                conn.rollback()
                return False, "Conta de destino inexistente."

            cursor.execute(
                "UPDATE usuarios SET saldo = saldo - ? WHERE numero_conta = ?",
                (valor, conta_origem)
            )
            cursor.execute(
                "UPDATE usuarios SET saldo = saldo + ? WHERE numero_conta = ?",
                (valor, conta_destino)
            )

            cursor.execute("""
                INSERT INTO transacoes (
                    conta_origem, conta_destino, valor, nivel_seguranca,
                    exigiu_biometria, liveness_aprovado, distancia_euclidiana,
                    similaridade_cosseno, latencia_ms, status, data_hora
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conta_origem,
                conta_destino,
                valor,
                nivel_seguranca,
                1 if exigiu_biometria else 0,
                1 if liveness_aprovado else 0,
                distancia_euclidiana,
                similaridade_cosseno,
                latencia_ms,
                "APROVADA",
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            ))

            conn.commit()
            return True, "Transferência concluída com sucesso!"
        except Exception as e:
            conn.rollback()
            return False, f"Erro transacional: {str(e)}"
        finally:
            conn.close()

    def registrar_tentativa_bloqueada(
        self,
        conta_origem: str,
        conta_destino: str,
        valor: float,
        nivel_seguranca: str,
        distancia_euclidiana: Optional[float],
        similaridade_cosseno: Optional[float],
        latencia_ms: Optional[float],
        motivo: str
    ):
        """Grava tentativas rejeitadas para auditoria forense e cálculo de métricas."""
        with self._conectar() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transacoes (
                    conta_origem, conta_destino, valor, nivel_seguranca,
                    exigiu_biometria, liveness_aprovado, distancia_euclidiana,
                    similaridade_cosseno, latencia_ms, status, data_hora
                ) VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?)
            """, (
                conta_origem,
                conta_destino,
                valor,
                nivel_seguranca,
                distancia_euclidiana,
                similaridade_cosseno,
                latencia_ms,
                f"BLOQUEADA ({motivo})",
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            ))
            conn.commit()

    def listar_extrato(self, numero_conta: str, filtro_status: str = "TODOS") -> List[sqlite3.Row]:
        with self._conectar() as conn:
            cursor = conn.cursor()
            if filtro_status == "APROVADA":
                cursor.execute("""
                    SELECT * FROM transacoes
                    WHERE (conta_origem = ? OR conta_destino = ?) AND status = 'APROVADA'
                    ORDER BY id DESC LIMIT 30
                """, (numero_conta, numero_conta))
            elif filtro_status == "BLOQUEADA":
                cursor.execute("""
                    SELECT * FROM transacoes
                    WHERE (conta_origem = ? OR conta_destino = ?) AND status LIKE 'BLOQUEADA%'
                    ORDER BY id DESC LIMIT 30
                """, (numero_conta, numero_conta))
            else:
                cursor.execute("""
                    SELECT * FROM transacoes
                    WHERE conta_origem = ? OR conta_destino = ?
                    ORDER BY id DESC LIMIT 30
                """, (numero_conta, numero_conta))
            return cursor.fetchall()

    def exportar_relatorio_auditoria_csv(self, numero_conta: str, caminho_arquivo: str) -> Tuple[bool, str]:
        """Exporta o extrato auditável completo em CSV para documentação da APS."""
        registros = self.listar_extrato(numero_conta, filtro_status="TODOS")
        try:
            with open(caminho_arquivo, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "ID", "Data_Hora", "Conta_Origem", "Conta_Destino", "Valor_BRL",
                    "Nivel_Seguranca", "Exigiu_Biometria", "Liveness_OK",
                    "Distancia_Euclidiana", "Similaridade_Cosseno", "Latencia_ms", "Status"
                ])
                for r in registros:
                    writer.writerow([
                        r["id"],
                        r["data_hora"],
                        r["conta_origem"],
                        r["conta_destino"],
                        f"{r['valor']:.2f}",
                        r["nivel_seguranca"],
                        "SIM" if r["exigiu_biometria"] else "NAO",
                        "SIM" if r["liveness_aprovado"] else "NAO",
                        f"{r['distancia_euclidiana']:.4f}" if r["distancia_euclidiana"] is not None else "",
                        f"{r['similaridade_cosseno']:.4f}" if r["similaridade_cosseno"] is not None else "",
                        f"{r['latencia_ms']:.2f}" if r["latencia_ms"] is not None else "",
                        r["status"]
                    ])
            return True, f"Relatório exportado com sucesso para:\n{caminho_arquivo}"
        except Exception as e:
            return False, f"Falha ao exportar relatório: {str(e)}"