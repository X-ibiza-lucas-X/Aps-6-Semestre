from __future__ import annotations

import base64
import csv
import hashlib
import io
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from numpy.typing import NDArray

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES DO SISTEMA (MINISTÉRIO DO MEIO AMBIENTE)
# ==============================================================================
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
STORAGE_DIR: Final[Path] = BASE_DIR / "dados_cofre_mma"
DATASET_DIR: Final[Path] = STORAGE_DIR / "dataset_faces"
MODEL_FILE: Final[Path] = STORAGE_DIR / "modelo_lbph_mma.yml"
DB_FILE: Final[Path] = STORAGE_DIR / "cofre_mma.db"

ROI_SIZE: Final[tuple[int, int]] = (200, 200)
CONFIDENCE_LIMIT: Final[float] = 65.0  # Quanto menor a distância LBPH, maior a precisão


class NivelAcesso(IntEnum):
    NEGADO = 0
    NIVEL_1_GERAL = 1
    NIVEL_2_DIRETOR = 2
    NIVEL_3_MINISTRO = 3

    @property
    def titulo(self) -> str:
        match self:
            case NivelAcesso.NIVEL_1_GERAL:
                return "Nível 1: Permissão Geral"
            case NivelAcesso.NIVEL_2_DIRETOR:
                return "Nível 2: Diretor de Divisão"
            case NivelAcesso.NIVEL_3_MINISTRO:
                return "Nível 3: Acesso Total (Ministro / Tesoureiro)"
            case _:
                return "Acesso Negado"


@dataclass(slots=True)
class EstadoAtualPIVC:
    """Armazena em tempo real o resultado das 5 fases do fluxograma PIVC."""
    fase1_b64: str = ""
    fase2_b64: str = ""
    fase3_b64: str = ""
    fase4_b64: str = ""
    status_acesso: str = "AGUARDANDO"
    usuario_nome: str = "Nenhum usuário autenticado"
    usuario_cargo: str = "-"
    nivel: int = 0
    nivel_descricao: str = "Acesso Bloqueado"
    confianca_percentual: float = 0.0
    distancia_lbph: float = 0.0
    mensagem: str = "Posicione o rosto diante do sensor ou autentique com usuário/senha."


# ==============================================================================
# BANCO DE DADOS RELACIONAL (SQLite) — USUÁRIOS, SENHAS E AUDITORIA
# ==============================================================================
class BancoCofreMMA:
    def __init__(self, db_path: Path) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._inicializar()
        self._popular_usuarios_iniciais()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def hash_senha(senha: str) -> str:
        return hashlib.sha256(senha.strip().encode("utf-8")).hexdigest()

    def _inicializar(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nome TEXT NOT NULL UNIQUE,
                    cargo TEXT NOT NULL,
                    nivel INTEGER NOT NULL CHECK(nivel IN (1, 2, 3)),
                    senha_hash TEXT NOT NULL,
                    biometria_ativa INTEGER NOT NULL DEFAULT 0,
                    origem_aquisicao TEXT NOT NULL,
                    criado_em TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relatorio_acessos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_hora TEXT NOT NULL,
                    usuario_id INTEGER,
                    nome TEXT NOT NULL,
                    nivel INTEGER NOT NULL,
                    confianca REAL NOT NULL,
                    resultado TEXT NOT NULL,
                    metodo TEXT NOT NULL
                );
                """
            )

            colunas = {row["name"] for row in c.execute("PRAGMA table_info(usuarios)").fetchall()}
            if "senha_hash" not in colunas:
                c.execute("ALTER TABLE usuarios ADD COLUMN senha_hash TEXT NOT NULL DEFAULT ''")
                senha_padrao = self.hash_senha("1234")
                c.execute("UPDATE usuarios SET senha_hash = ? WHERE senha_hash = ''", (senha_padrao,))
            if "biometria_ativa" not in colunas:
                c.execute("ALTER TABLE usuarios ADD COLUMN biometria_ativa INTEGER NOT NULL DEFAULT 0")

    def _popular_usuarios_iniciais(self) -> None:
        """Popula o SQLite com Lucas (Tesoureiro - acesso total) e demais usuários."""
        with self._lock, self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0]
            if total > 0:
                return

            agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            senha_padrao = self.hash_senha("1234")

            # Níveis alternativos predefinidos para Igor, Gustavo, Moisés e Guilherme
            perfis_alternativos = [
                ("IGOR", "Diretor de Divisão de Contenção", 2),
                ("GUSTAVO", "Analista Ambiental", 1),
                ("MOISÉS", "Diretor de Divisão Química", 2),
                ("GUILHERME", "Pesquisador de Campo", 1),
            ]
            random.shuffle(perfis_alternativos)

            usuarios_seed = [
                ("LUCAS", "Tesoureiro (Acesso Total aos Cofres)", 3, senha_padrao, 0, "Seed SQLite", agora),
                *[
                    (nome, cargo, nivel, senha_padrao, 0, "Seed SQLite", agora)
                    for nome, cargo, nivel in perfis_alternativos
                ],
            ]

            c.executemany(
                """
                INSERT INTO usuarios (nome, cargo, nivel, senha_hash, biometria_ativa, origem_aquisicao, criado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                usuarios_seed,
            )

    def criar_ou_atualizar_usuario(
        self, nome: str, cargo: str, nivel: int, senha: str, origem: str, biometria_ativa: int = 0
    ) -> int:
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nome_upper = nome.strip().upper()
        senha_h = self.hash_senha(senha or "1234")

        with self._lock, self._conn() as c:
            existente = c.execute("SELECT id FROM usuarios WHERE nome = ?", (nome_upper,)).fetchone()
            if existente:
                uid = int(existente["id"])
                c.execute(
                    """
                    UPDATE usuarios
                    SET cargo = ?, nivel = ?, senha_hash = ?, biometria_ativa = ?, origem_aquisicao = ?
                    WHERE id = ?
                    """,
                    (cargo.strip(), nivel, senha_h, biometria_ativa, origem, uid),
                )
                return uid

            cur = c.execute(
                """
                INSERT INTO usuarios (nome, cargo, nivel, senha_hash, biometria_ativa, origem_aquisicao, criado_em)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (nome_upper, cargo.strip(), nivel, senha_h, biometria_ativa, origem, agora),
            )
            return int(cur.lastrowid)

    def marcar_biometria_ativa(self, user_id: int, origem: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE usuarios SET biometria_ativa = 1, origem_aquisicao = ? WHERE id = ?",
                (origem, user_id),
            )

    def autenticar_por_senha(self, nome: str, senha: str) -> dict | None:
        nome_upper = nome.strip().upper()
        senha_h = self.hash_senha(senha)
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT * FROM usuarios WHERE nome = ? AND senha_hash = ?",
                (nome_upper, senha_h),
            ).fetchone()
        return dict(row) if row else None

    def listar_usuarios(self) -> dict[int, dict]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT * FROM usuarios ORDER BY id ASC").fetchall()
        return {int(r["id"]): dict(r) for r in rows}

    def excluir_usuario(self, user_id: int) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM usuarios WHERE id = ?", (user_id,))
        pasta = DATASET_DIR / f"user_{user_id}"
        if pasta.exists():
            for f in pasta.glob("*"):
                f.unlink()
            pasta.rmdir()

    def registrar_log(
        self, user_id: int | None, nome: str, nivel: int, confianca: float, resultado: str, metodo: str
    ) -> None:
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO relatorio_acessos (data_hora, usuario_id, nome, nivel, confianca, resultado, metodo)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (agora, user_id, nome, nivel, round(confianca, 2), resultado, metodo),
            )

    def listar_logs(self, limite: int = 50) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT * FROM relatorio_acessos ORDER BY id DESC LIMIT ?", (limite,)).fetchall()
        return [dict(r) for r in rows]


# ==============================================================================
# MOTOR DE PROCESSAMENTO DE IMAGENS DIGITAIS (AS 5 FASES DO MANUAL PIVC)
# ==============================================================================
class MotorPIVC:
    def __init__(self, banco: BancoCofreMMA) -> None:
        self.banco = banco
        self.lock_modelo = threading.Lock()

        haar_dir = Path(cv2.data.haarcascades)
        self.face_detector = cv2.CascadeClassifier(str(haar_dir / "haarcascade_frontalface_default.xml"))
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        self.reconhecedor = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
        self.treinado = False
        self.recarregar_modelo()

    def recarregar_modelo(self) -> bool:
        with self.lock_modelo:
            if MODEL_FILE.exists():
                self.reconhecedor.read(str(MODEL_FILE))
                self.treinado = True
                return True
            self.treinado = False
            return False

    # FASE 2: PRÉ-PROCESSAMENTO
    def fase2_preprocessar(self, img_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        filtro_gaussiano = cv2.GaussianBlur(cinza, (5, 5), 0)
        return self.clahe.apply(filtro_gaussiano)

    # FASE 3: SEGMENTAÇÃO (Detecção Facial + Extração da ROI)
    def fase3_segmentar(
        self, img_prep: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8] | None, tuple[int, int, int, int] | None]:
        faces = self.face_detector.detectMultiScale(
            img_prep, scaleFactor=1.15, minNeighbors=5, minSize=(90, 90)
        )
        if len(faces) == 0:
            return None, None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = img_prep[y : y + h, x : x + w]
        roi_padronizada = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_CUBIC)
        return roi_padronizada, (int(x), int(y), int(w), int(h))

    # FASE 4: EXTRAÇÃO DE CARACTERÍSTICAS (Mapa Visual LBP + Treinamento LBPH)
    @staticmethod
    def fase4_gerar_mapa_lbp(roi_gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
        img = roi_gray.astype(np.int16)
        centro = img[1:-1, 1:-1]
        lbp = np.zeros_like(centro, dtype=np.uint8)
        vizinhos = [
            (-1, -1, 128), (-1, 0, 64), (-1, 1, 32),
            (0, 1, 16), (1, 1, 8), (1, 0, 4),
            (1, -1, 2), (0, -1, 1),
        ]
        for dy, dx, peso in vizinhos:
            deslocada = img[1 + dy : img.shape[0] - 1 + dy, 1 + dx : img.shape[1] - 1 + dx]
            lbp |= ((deslocada >= centro).astype(np.uint8) * peso)
        return cv2.resize(lbp, ROI_SIZE, interpolation=cv2.INTER_NEAREST)

    def treinar_base_completa(self) -> bool:
        faces_treino: list[NDArray[np.uint8]] = []
        ids_treino: list[int] = []
        usuarios_validos = self.banco.listar_usuarios()

        for pasta in DATASET_DIR.iterdir():
            if not pasta.is_dir() or not pasta.name.startswith("user_"):
                continue
            uid = int(pasta.name.split("_")[1])
            if uid not in usuarios_validos:
                continue
            for arq in pasta.glob("*.png"):
                img = cv2.imread(str(arq), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces_treino.append(img)
                    ids_treino.append(uid)

        with self.lock_modelo:
            if not faces_treino:
                if MODEL_FILE.exists():
                    MODEL_FILE.unlink()
                self.treinado = False
                return False

            self.reconhecedor = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            self.reconhecedor.train(faces_treino, np.array(ids_treino, dtype=np.int32))
            self.reconhecedor.write(str(MODEL_FILE))
            self.treinado = True
            return True

    # FASE 5: CLASSIFICAÇÃO E INTERPRETAÇÃO
    def fase5_classificar(self, roi_gray: NDArray[np.uint8]) -> tuple[dict | None, float, float]:
        usuarios = self.banco.listar_usuarios()
        with self.lock_modelo:
            if not self.treinado or not usuarios:
                return None, 999.0, 0.0
            uid_predito, distancia = self.reconhecedor.predict(roi_gray)

        confianca = max(0.0, min(100.0, (1.0 - (distancia / 120.0)) * 100.0))
        if distancia <= CONFIDENCE_LIMIT and uid_predito in usuarios:
            return usuarios[uid_predito], float(distancia), round(confianca, 1)
        return None, float(distancia), round(confianca, 1)


# ==============================================================================
# GERENCIADOR DE SENSOR ÓPTICO (WEBCAM EM TEMPO REAL + PIPELINE CONTÍNUO)
# ==============================================================================
class GerenciadorSensorCofre:
    def __init__(self, motor: MotorPIVC, banco: BancoCofreMMA) -> None:
        self.motor = motor
        self.banco = banco
        self.cap: cv2.VideoCapture | None = None
        self.ativo = False
        self.estado = EstadoAtualPIVC()
        self.frame_atual_jpeg: bytes | None = None
        self._ultimo_log_ts = 0.0

        self.coletando_usuario_id: int | None = None
        self.amostras_coletadas = 0
        self.meta_amostras = 25

    @staticmethod
    def _img_to_b64(img: NDArray[np.uint8], largura: int = 180) -> str:
        h, w = img.shape[:2]
        nova_h = int(h * (largura / w))
        mini = cv2.resize(img, (largura, nova_h))
        _, buf = cv2.imencode(".jpg", mini, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buf).decode("ascii")

    def iniciar(self) -> tuple[bool, str]:
        if self.ativo:
            return True, "Sensor já está ativo."
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False, "Nenhuma webcam encontrada. Utilize o Login por Senha ou Scanner/Arquivo."
        self.ativo = True
        threading.Thread(target=self._loop_processamento, daemon=True).start()
        return True, "Sensor óptico iniciado com sucesso."

    def parar(self) -> None:
        self.ativo = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _loop_processamento(self) -> None:
        while self.ativo and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            self._processar_unico_frame(frame, origem="Webcam Tempo Real")
            time.sleep(0.02)

    def _processar_unico_frame(self, frame_bgr: NDArray[np.uint8], origem: str) -> EstadoAtualPIVC:
        hud = frame_bgr.copy()
        f1_b64 = self._img_to_b64(frame_bgr, 180)

        img_prep = self.motor.fase2_preprocessar(frame_bgr)
        f2_b64 = self._img_to_b64(img_prep, 180)

        roi, bbox = self.motor.fase3_segmentar(img_prep)
        novo_estado = EstadoAtualPIVC(fase1_b64=f1_b64, fase2_b64=f2_b64)

        if roi is not None and bbox is not None:
            x, y, w, h = bbox
            novo_estado.fase3_b64 = self._img_to_b64(roi, 150)

            mapa_lbp = self.motor.fase4_gerar_mapa_lbp(roi)
            novo_estado.fase4_b64 = self._img_to_b64(mapa_lbp, 150)

            if self.coletando_usuario_id is not None:
                pasta = DATASET_DIR / f"user_{self.coletando_usuario_id}"
                pasta.mkdir(parents=True, exist_ok=True)
                self.amostras_coletadas += 1
                cv2.imwrite(str(pasta / f"amostra_{self.amostras_coletadas:03d}.png"), roi)
                cv2.rectangle(hud, (x, y), (x + w, y + h), (255, 200, 0), 3)
                cv2.putText(
                    hud,
                    f"COLETANDO BIOMETRIA: {self.amostras_coletadas}/{self.meta_amostras}",
                    (x, max(25, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 200, 0),
                    2,
                )
                if self.amostras_coletadas >= self.meta_amostras:
                    uid_concluido = self.coletando_usuario_id
                    self.coletando_usuario_id = None
                    self.banco.marcar_biometria_ativa(uid_concluido, "Webcam Tempo Real")
                    self.motor.treinar_base_completa()

            usuario, distancia, confianca = self.motor.fase5_classificar(roi)
            novo_estado.distancia_lbph = round(distancia, 2)
            novo_estado.confianca_percentual = confianca

            if usuario is not None:
                nivel_enum = NivelAcesso(usuario["nivel"])
                cor = (0, 230, 118) if nivel_enum == NivelAcesso.NIVEL_3_MINISTRO else (0, 191, 255)
                novo_estado.status_acesso = "PERMITIDO"
                novo_estado.usuario_nome = usuario["nome"]
                novo_estado.usuario_cargo = usuario["cargo"]
                novo_estado.nivel = int(nivel_enum)
                novo_estado.nivel_descricao = nivel_enum.titulo
                novo_estado.mensagem = f"Autenticado via Biometria ({confianca}%)."

                cv2.rectangle(hud, (x, y), (x + w, y + h), cor, 3)
                cv2.rectangle(hud, (x, y - 55), (x + max(w, 360), y), cor, -1)
                cv2.putText(hud, usuario["nome"], (x + 8, y - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                cv2.putText(hud, nivel_enum.titulo, (x + 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)

                if time.time() - self._ultimo_log_ts > 4.0 and self.coletando_usuario_id is None:
                    self.banco.registrar_log(
                        usuario["id"], usuario["nome"], usuario["nivel"], confianca, "PERMITIDO", origem
                    )
                    self._ultimo_log_ts = time.time()
            else:
                cor = (0, 0, 255)
                novo_estado.status_acesso = "NEGADO"
                novo_estado.usuario_nome = "INDIVIDUO NAO IDENTIFICADO"
                novo_estado.nivel = 0
                novo_estado.nivel_descricao = "Acesso Negado ao Cofre"
                novo_estado.mensagem = "Rosto não reconhecido ou sem biometria vinculada."

                cv2.rectangle(hud, (x, y), (x + w, y + h), cor, 3)
                cv2.rectangle(hud, (x, y - 38), (x + max(w, 300), y), cor, -1)
                cv2.putText(hud, "ACESSO NEGADO", (x + 8, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                if time.time() - self._ultimo_log_ts > 4.0 and self.coletando_usuario_id is None:
                    self.banco.registrar_log(None, "DESCONHECIDO", 0, confianca, "NEGADO", origem)
                    self._ultimo_log_ts = time.time()

        _, jpeg = cv2.imencode(".jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        self.frame_atual_jpeg = jpeg.tobytes()
        self.estado = novo_estado
        return novo_estado


# ==============================================================================
# APLICAÇÃO WEB INTERATIVA (FLASK + INTERFACE HTML5 RESPONSIVA)
# ==============================================================================
app = Flask(__name__)
banco = BancoCofreMMA(DB_FILE)
motor = MotorPIVC(banco)
sensor = GerenciadorSensorCofre(motor, banco)

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cofre Biométrico — Ministério do Meio Ambiente (PIVC)</title>
<style>
  :root {
    --bg: #0b1120; --card: #111827; --border: #1f2937;
    --primary: #10b981; --accent: #38bdf8; --danger: #ef4444; --warn: #f59e0b;
    --text: #f3f4f6; --muted: #9ca3af;
  }
  * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); }
  header {
    background: linear-gradient(90deg, #064e3b, #0f172a);
    padding: 16px 28px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
  }
  header h1 { margin: 0; font-size: 1.25rem; letter-spacing: 0.5px; }
  header small { color: #a7f3d0; display: block; font-size: 0.82rem; }
  .tabs { display: flex; gap: 8px; flex-wrap: wrap; }
  .tab-btn {
    background: rgba(255,255,255,0.08); color: var(--text); border: 1px solid rgba(255,255,255,0.15);
    padding: 9px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: 0.2s;
  }
  .tab-btn.active, .tab-btn:hover { background: var(--primary); color: #052e16; border-color: var(--primary); }
  .container { max-width: 1400px; margin: 20px auto; padding: 0 20px; }
  .grid-main { display: grid; grid-template-columns: 1.35fr 1fr; gap: 20px; }
  @media (max-width: 980px) { .grid-main { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .card h2 { margin-top: 0; font-size: 1.05rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 10px; }
  .video-box {
    width: 100%; aspect-ratio: 16/9; background: #030712; border-radius: 10px;
    display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1px solid var(--border);
  }
  .video-box img { width: 100%; height: 100%; object-fit: contain; }
  .controls { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
  button.btn, label.btn, a.btn {
    padding: 10px 16px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.88rem; display: inline-block;
  }
  .btn-green { background: var(--primary); color: #052e16; }
  .btn-red { background: var(--danger); color: white; }
  .btn-blue { background: var(--accent); color: #082f49; }
  .btn-warn { background: var(--warn); color: #451a03; }
  .status-banner {
    padding: 15px; border-radius: 10px; margin-bottom: 14px; font-weight: 700; text-align: center;
    background: #1f2937; border: 2px solid var(--border); transition: 0.3s;
  }
  .status-PERMITIDO { background: rgba(16,185,129,0.15); border-color: var(--primary); color: #6ee7b7; }
  .status-NEGADO { background: rgba(239,68,68,0.15); border-color: var(--danger); color: #fca5a5; }
  .pivc-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
  .pivc-step { background: #030712; border: 1px solid var(--border); border-radius: 8px; padding: 6px; text-align: center; }
  .pivc-step span { display: block; font-size: 0.72rem; color: var(--muted); margin-bottom: 4px; }
  .pivc-step img { width: 100%; height: 85px; object-fit: cover; border-radius: 4px; background: #111827; }
  .vault-levels { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
  .vault-box { padding: 14px; border-radius: 10px; border: 1px solid var(--border); background: #030712; opacity: 0.4; transition: 0.3s; }
  .vault-box.unlocked { opacity: 1; border-color: var(--primary); background: rgba(16,185,129,0.1); box-shadow: 0 0 15px rgba(16,185,129,0.15); }
  .vault-box h4 { margin: 0 0 6px 0; font-size: 0.92rem; }
  .vault-box p { margin: 0; font-size: 0.80rem; color: var(--muted); }
  .form-group { margin-bottom: 12px; }
  .form-group label { display: block; font-size: 0.85rem; margin-bottom: 5px; color: var(--muted); }
  .form-group input, .form-group select {
    width: 100%; padding: 10px; border-radius: 8px; border: 1px solid var(--border);
    background: #030712; color: var(--text);
  }
  .login-box {
    background: #030712; border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-top: 16px;
  }
  .login-row { display: grid; grid-template-columns: 1fr 1fr auto; gap: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  th, td { padding: 9px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--accent); }
  .badge { padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
  .badge-ok { background: rgba(16,185,129,0.2); color: #6ee7b7; }
  .badge-no { background: rgba(245,158,11,0.2); color: #fcd34d; }
  .hidden { display: none; }
</style>
</head>
<body>

<header>
  <div>
    <h1>🛡️ MINISTÉRIO DO MEIO AMBIENTE — COFRE DE TOXINAS CRÍTICAS</h1>
    <small>Sistema de Identificação e Autenticação Biométrica + Banco SQLite (5 Fases PIVC)</small>
  </div>
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('tab-cofre', this)">1. Controle do Cofre</button>
    <button class="tab-btn" onclick="showTab('tab-cadastro', this)">2. Usuários & Biometria (SQLite)</button>
    <button class="tab-btn" onclick="showTab('tab-relatorios', this)">3. Relatórios de Auditoria</button>
  </div>
</header>

<div class="container">
  <!-- ABA 1: MONITORAMENTO E COFRE -->
  <div id="tab-cofre" class="grid-main">
    <div class="card">
      <h2>Fase 1: Sensor (Aquisição em Tempo Real ou Scanner)</h2>
      <div class="video-box">
        <img id="stream-img" src="/video_feed" alt="Sensor óptico em standby...">
      </div>
      <div class="controls">
        <button class="btn btn-green" onclick="controlarCamera('start')">▶ Ligar Câmera (Tempo Real)</button>
        <button class="btn btn-red" onclick="controlarCamera('stop')">⏹ Desligar Câmera</button>
        <label class="btn btn-blue">
          🖼️ Testar Imagem de Scanner/Arquivo
          <input type="file" accept="image/*" style="display:none" onchange="analisarArquivo(this)">
        </label>
      </div>

      <h2 style="margin-top:18px;">Pipeline das 5 Fases do Processamento de Imagens Digitais</h2>
      <div class="pivc-grid">
        <div class="pivc-step"><span>1. Sensor (Bruta)</span><img id="img-f1"></div>
        <div class="pivc-step"><span>2. Pré-Proc. (CLAHE)</span><img id="img-f2"></div>
        <div class="pivc-step"><span>3. Segmentação (ROI)</span><img id="img-f3"></div>
        <div class="pivc-step"><span>4. Extração (LBP)</span><img id="img-f4"></div>
      </div>
    </div>

    <div class="card">
      <h2>Fase 5: Classificação, Interpretação e Permissões do Cofre</h2>
      <div id="status-banner" class="status-banner">AGUARDANDO AUTENTICAÇÃO BIOMÉTRICA OU CREDENCIAL</div>

      <p><strong>Usuário Identificado:</strong> <span id="lbl-nome">-</span></p>
      <p><strong>Cargo / Função:</strong> <span id="lbl-cargo">-</span></p>
      <p><strong>Nível de Segurança:</strong> <span id="lbl-nivel">Bloqueado</span></p>
      <p><strong>Confiança / Método:</strong> <span id="lbl-conf">0%</span> (Distância LBPH: <span id="lbl-dist">-</span>)</p>

      <div class="login-box">
        <strong style="display:block; margin-bottom:8px; color:var(--accent); font-size:0.88rem;">
          🔑 Autenticação Direta SQLite (Usuário + Senha 1234)
        </strong>
        <div class="login-row">
          <select id="login-usuario"></select>
          <input type="password" id="login-senha" placeholder="Senha (1234)" value="1234">
          <button class="btn btn-warn" onclick="autenticarPorSenha()">Entrar</button>
        </div>
      </div>

      <h2 style="margin-top:18px;">Compartimentos do Cofre de Toxinas (3 Níveis)</h2>
      <div class="vault-levels">
        <div id="box-n1" class="vault-box">
          <h4>🔓 Nível 1: Geral</h4>
          <p>Protocolos básicos de contenção ambiental e normas gerais do MMA.</p>
        </div>
        <div id="box-n2" class="vault-box">
          <h4>🔓 Nível 2: Diretores</h4>
          <p>Relatórios técnicos das divisões específicas e inventário de agentes químicos.</p>
        </div>
        <div id="box-n3" class="vault-box">
          <h4>🔓 Nível 3: Total (Ministro / Tesoureiro)</h4>
          <p>Acesso irrestrito a todos os cofres e toxinas de risco pandêmico global.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- ABA 2: CADASTRO BIOMÉTRICO -->
  <div id="tab-cadastro" class="grid-main hidden">
    <div class="card">
      <h2>Cadastrar ou Vincular Biometria a Usuário SQLite</h2>
      <div class="form-group">
        <label>Nome do Usuário (Ex: LUCAS, IGOR, GUSTAVO, MOISÉS, GUILHERME ou novo):</label>
        <input type="text" id="cad-nome" placeholder="LUCAS">
      </div>
      <div class="form-group">
        <label>Cargo / Divisão:</label>
        <input type="text" id="cad-cargo" placeholder="Tesoureiro">
      </div>
      <div class="form-group">
        <label>Senha de Acesso:</label>
        <input type="password" id="cad-senha" value="1234">
      </div>
      <div class="form-group">
        <label>Nível de Permissão (Conforme Manual):</label>
        <select id="cad-nivel">
          <option value="1">Nível 1 — Acesso concedido a indivíduos com permissão geral</option>
          <option value="2">Nível 2 — Acesso restrito aos diretores de divisões específicas</option>
          <option value="3" selected>Nível 3 — Acesso total a todos os cofres (Ministro / Tesoureiro)</option>
        </select>
      </div>
      <div class="controls">
        <button class="btn btn-green" onclick="cadastrarViaWebcam()">📸 Capturar 25 Amostras pela Webcam</button>
        <label class="btn btn-blue">
          📄 Vincular Foto / Scanner
          <input type="file" accept="image/*" style="display:none" onchange="cadastrarViaArquivo(this)">
        </label>
      </div>
      <p id="cad-feedback" style="color:var(--accent); font-weight:600; margin-top:14px;"></p>
    </div>

    <div class="card">
      <h2>Banco de Dados SQLite (cofre_mma.db)</h2>
      <table>
        <thead><tr><th>ID</th><th>Nome</th><th>Cargo</th><th>Nível</th><th>Biometria</th><th>Ação</th></tr></thead>
        <tbody id="tabela-usuarios"></tbody>
      </table>
    </div>
  </div>

  <!-- ABA 3: RELATÓRIOS -->
  <div id="tab-relatorios" class="card hidden">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
      <h2 style="border:none; margin:0;">Relatório de Auditoria de Acessos ao Cofre</h2>
      <a href="/exportar_csv" class="btn btn-green" style="text-decoration:none;">📥 Baixar Relatório CSV (Excel)</a>
    </div>
    <table>
      <thead><tr><th>Data/Hora</th><th>Usuário</th><th>Nível</th><th>Confiança</th><th>Resultado</th><th>Método</th></tr></thead>
      <tbody id="tabela-logs"></tbody>
    </table>
  </div>
</div>

<script>
function showTab(id, btn) {
  document.querySelectorAll('#tab-cofre, #tab-cadastro, #tab-relatorios').forEach(el => el.classList.add('hidden'));
  document.getElementById(id).classList.remove('hidden');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  carregarUsuarios();
  if (id === 'tab-relatorios') carregarLogs();
}

async function controlarCamera(acao) {
  const res = await fetch('/camera/' + acao, {method: 'POST'});
  const data = await res.json();
  if (!data.ok) alert(data.msg);
  if (acao === 'start') {
    document.getElementById('stream-img').src = '/video_feed?' + Date.now();
  }
}

async function autenticarPorSenha() {
  const nome = document.getElementById('login-usuario').value;
  const senha = document.getElementById('login-senha').value;
  const res = await fetch('/login_senha', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({nome, senha})
  });
  const data = await res.json();
  if (!data.ok) return alert(data.msg);
  atualizarPainelPIVC(data.estado);
}

async function analisarArquivo(input) {
  if (!input.files.length) return;
  const fd = new FormData();
  fd.append('imagem', input.files[0]);
  const res = await fetch('/analisar_imagem', {method: 'POST', body: fd});
  const data = await res.json();
  if (data.frame_b64) {
    document.getElementById('stream-img').src = 'data:image/jpeg;base64,' + data.frame_b64;
  }
  atualizarPainelPIVC(data.estado);
}

async function cadastrarViaWebcam() {
  const nome = document.getElementById('cad-nome').value.trim();
  const cargo = document.getElementById('cad-cargo').value.trim();
  const senha = document.getElementById('cad-senha').value.trim();
  const nivel = document.getElementById('cad-nivel').value;
  if (!nome || !cargo) return alert('Preencha Nome e Cargo!');

  const res = await fetch('/cadastrar_webcam', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({nome, cargo, senha, nivel})
  });
  const data = await res.json();
  document.getElementById('cad-feedback').innerText = data.msg;
  if (data.ok) {
    showTab('tab-cofre', document.querySelector('.tab-btn'));
  }
}

async function cadastrarViaArquivo(input) {
  const nome = document.getElementById('cad-nome').value.trim();
  const cargo = document.getElementById('cad-cargo').value.trim();
  const senha = document.getElementById('cad-senha').value.trim();
  const nivel = document.getElementById('cad-nivel').value;
  if (!nome || !cargo) return alert('Preencha Nome e Cargo antes de enviar a foto!');
  if (!input.files.length) return;

  const fd = new FormData();
  fd.append('nome', nome);
  fd.append('cargo', cargo);
  fd.append('senha', senha);
  fd.append('nivel', nivel);
  fd.append('imagem', input.files[0]);

  const res = await fetch('/cadastrar_arquivo', {method: 'POST', body: fd});
  const data = await res.json();
  document.getElementById('cad-feedback').innerText = data.msg;
  carregarUsuarios();
}

function preencherFormulario(nome, cargo, nivel) {
  document.getElementById('cad-nome').value = nome;
  document.getElementById('cad-cargo').value = cargo;
  document.getElementById('cad-nivel').value = nivel;
}

async function carregarUsuarios() {
  const res = await fetch('/usuarios');
  const lista = await res.json();

  const selectLogin = document.getElementById('login-usuario');
  selectLogin.innerHTML = lista.map(u => `<option value="${u.nome}">${u.nome} (Nível ${u.nivel})</option>`).join('');

  const tbody = document.getElementById('tabela-usuarios');
  tbody.innerHTML = lista.map(u => `
    <tr>
      <td>${u.id}</td>
      <td><strong>${u.nome}</strong></td>
      <td>${u.cargo}</td>
      <td>Nível ${u.nivel}</td>
      <td>${u.biometria_ativa ? '<span class="badge badge-ok">Rosto Ativo</span>' : '<span class="badge badge-no">Só Senha</span>'}</td>
      <td>
        <button class="btn btn-blue" style="padding:4px 8px" onclick="preencherFormulario('${u.nome}', '${u.cargo}', ${u.nivel})">Selecionar</button>
        <button class="btn btn-red" style="padding:4px 8px" onclick="excluirUsuario(${u.id})">Excluir</button>
      </td>
    </tr>
  `).join('');
}

async function excluirUsuario(id) {
  await fetch('/usuarios/' + id, {method: 'DELETE'});
  carregarUsuarios();
}

async function carregarLogs() {
  const res = await fetch('/logs');
  const lista = await res.json();
  document.getElementById('tabela-logs').innerHTML = lista.map(l => `
    <tr>
      <td>${l.data_hora}</td><td>${l.nome}</td><td>Nível ${l.nivel}</td>
      <td>${l.confianca}%</td><td><strong>${l.resultado}</strong></td><td>${l.metodo}</td>
    </tr>
  `).join('');
}

function atualizarPainelPIVC(st) {
  if (!st) return;
  if (st.fase1_b64) document.getElementById('img-f1').src = 'data:image/jpeg;base64,' + st.fase1_b64;
  if (st.fase2_b64) document.getElementById('img-f2').src = 'data:image/jpeg;base64,' + st.fase2_b64;
  if (st.fase3_b64) document.getElementById('img-f3').src = 'data:image/jpeg;base64,' + st.fase3_b64;
  if (st.fase4_b64) document.getElementById('img-f4').src = 'data:image/jpeg;base64,' + st.fase4_b64;

  const banner = document.getElementById('status-banner');
  banner.className = 'status-banner status-' + st.status_acesso;
  banner.innerText = st.status_acesso === 'PERMITIDO'
    ? `✅ ACESSO PERMITIDO — ${st.nivel_descricao.toUpperCase()}`
    : (st.status_acesso === 'NEGADO' ? '🚨 ACESSO NEGADO AO COFRE' : 'AGUARDANDO AUTENTICAÇÃO');

  document.getElementById('lbl-nome').innerText = st.usuario_nome;
  document.getElementById('lbl-cargo').innerText = st.usuario_cargo;
  document.getElementById('lbl-nivel').innerText = st.nivel_descricao;
  document.getElementById('lbl-conf').innerText = st.confianca_percentual + '%';
  document.getElementById('lbl-dist').innerText = st.distancia_lbph;

  const niv = st.status_acesso === 'PERMITIDO' ? st.nivel : 0;
  document.getElementById('box-n1').classList.toggle('unlocked', niv >= 1);
  document.getElementById('box-n2').classList.toggle('unlocked', niv >= 2);
  document.getElementById('box-n3').classList.toggle('unlocked', niv >= 3);
}

carregarUsuarios();

setInterval(async () => {
  try {
    const res = await fetch('/telemetria');
    const st = await res.json();
    if (st.status_acesso !== 'AGUARDANDO') atualizarPainelPIVC(st);
  } catch (e) {}
}, 400);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_INTERFACE)


@app.route("/camera/<acao>", methods=["POST"])
def controlar_camera(acao: str):
    if acao == "start":
        ok, msg = sensor.iniciar()
        return jsonify({"ok": ok, "msg": msg})
    sensor.parar()
    return jsonify({"ok": True, "msg": "Câmera desligada."})


@app.route("/video_feed")
def video_feed():
    def gerar_frames():
        while sensor.ativo:
            if sensor.frame_atual_jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + sensor.frame_atual_jpeg + b"\r\n"
                )
            time.sleep(0.03)

    return Response(gerar_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetria")
def telemetria():
    st = sensor.estado
    return jsonify(
        {
            "fase1_b64": st.fase1_b64,
            "fase2_b64": st.fase2_b64,
            "fase3_b64": st.fase3_b64,
            "fase4_b64": st.fase4_b64,
            "status_acesso": st.status_acesso,
            "usuario_nome": st.usuario_nome,
            "usuario_cargo": st.usuario_cargo,
            "nivel": st.nivel,
            "nivel_descricao": st.nivel_descricao,
            "confianca_percentual": st.confianca_percentual,
            "distancia_lbph": st.distancia_lbph,
        }
    )


@app.route("/login_senha", methods=["POST"])
def login_senha():
    dados = request.json or {}
    nome = str(dados.get("nome", "")).strip()
    senha = str(dados.get("senha", "")).strip()

    usuario = banco.autenticar_por_senha(nome, senha)
    if not usuario:
        banco.registrar_log(None, nome.upper() or "DESCONHECIDO", 0, 0.0, "NEGADO", "Senha SQLite")
        return jsonify({"ok": False, "msg": "Credenciais inválidas! Verifique o usuário e a senha."})

    nivel_enum = NivelAcesso(usuario["nivel"])
    banco.registrar_log(usuario["id"], usuario["nome"], usuario["nivel"], 100.0, "PERMITIDO", "Senha SQLite")

    sensor.estado = EstadoAtualPIVC(
        status_acesso="PERMITIDO",
        usuario_nome=usuario["nome"],
        usuario_cargo=usuario["cargo"],
        nivel=int(nivel_enum),
        nivel_descricao=nivel_enum.titulo,
        confianca_percentual=100.0,
        distancia_lbph=0.0,
        mensagem="Autenticado via credencial SQLite.",
    )
    return jsonify(
        {
            "ok": True,
            "estado": {
                "status_acesso": sensor.estado.status_acesso,
                "usuario_nome": sensor.estado.usuario_nome,
                "usuario_cargo": sensor.estado.usuario_cargo,
                "nivel": sensor.estado.nivel,
                "nivel_descricao": sensor.estado.nivel_descricao,
                "confianca_percentual": 100.0,
                "distancia_lbph": 0.0,
            },
        }
    )


@app.route("/analisar_imagem", methods=["POST"])
def analisar_imagem():
    arquivo = request.files.get("imagem")
    if not arquivo:
        return jsonify({"ok": False})
    bytes_img = np.frombuffer(arquivo.read(), np.uint8)
    img_bgr = cv2.imdecode(bytes_img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"ok": False})

    estado = sensor._processar_unico_frame(img_bgr, origem="Scanner / Arquivo")
    frame_b64 = base64.b64encode(sensor.frame_atual_jpeg).decode("ascii") if sensor.frame_atual_jpeg else ""
    return jsonify(
        {
            "ok": True,
            "frame_b64": frame_b64,
            "estado": {
                "fase1_b64": estado.fase1_b64,
                "fase2_b64": estado.fase2_b64,
                "fase3_b64": estado.fase3_b64,
                "fase4_b64": estado.fase4_b64,
                "status_acesso": estado.status_acesso,
                "usuario_nome": estado.usuario_nome,
                "usuario_cargo": estado.usuario_cargo,
                "nivel": estado.nivel,
                "nivel_descricao": estado.nivel_descricao,
                "confianca_percentual": estado.confianca_percentual,
                "distancia_lbph": estado.distancia_lbph,
            },
        }
    )


@app.route("/cadastrar_webcam", methods=["POST"])
def cadastrar_webcam():
    dados = request.json or {}
    nome = str(dados.get("nome", "")).strip()
    cargo = str(dados.get("cargo", "")).strip()
    senha = str(dados.get("senha", "1234")).strip()
    nivel = int(dados.get("nivel", 1))
    if not nome or not cargo:
        return jsonify({"ok": False, "msg": "Informe nome e cargo."})

    if not sensor.ativo:
        ok, msg = sensor.iniciar()
        if not ok:
            return jsonify({"ok": False, "msg": msg})

    uid = banco.criar_ou_atualizar_usuario(nome, cargo, nivel, senha, "Webcam Tempo Real", biometria_ativa=0)
    sensor.amostras_coletadas = 0
    sensor.coletando_usuario_id = uid
    return jsonify({"ok": True, "msg": f"Olhe para a câmera! Coletando {sensor.meta_amostras} amostras para {nome}..."})


@app.route("/cadastrar_arquivo", methods=["POST"])
def cadastrar_arquivo():
    nome = request.form.get("nome", "").strip()
    cargo = request.form.get("cargo", "").strip()
    senha = request.form.get("senha", "1234").strip()
    nivel = int(request.form.get("nivel", 1))
    arquivo = request.files.get("imagem")
    if not nome or not cargo or not arquivo:
        return jsonify({"ok": False, "msg": "Preencha todos os campos e selecione uma imagem."})

    img_bgr = cv2.imdecode(np.frombuffer(arquivo.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"ok": False, "msg": "Arquivo de imagem inválido."})

    prep = motor.fase2_preprocessar(img_bgr)
    roi, _ = motor.fase3_segmentar(prep)
    if roi is None:
        return jsonify({"ok": False, "msg": "Nenhum rosto foi detectado na imagem enviada."})

    uid = banco.criar_ou_atualizar_usuario(nome, cargo, nivel, senha, "Scanner / Arquivo", biometria_ativa=1)
    pasta = DATASET_DIR / f"user_{uid}"
    pasta.mkdir(parents=True, exist_ok=True)
    for idx, alfa in enumerate([0.92, 0.96, 1.0, 1.04, 1.08], start=1):
        ajustada = np.clip(roi.astype(np.float32) * alfa, 0, 255).astype(np.uint8)
        cv2.imwrite(str(pasta / f"scan_{idx:02d}.png"), ajustada)

    motor.treinar_base_completa()
    return jsonify({"ok": True, "msg": f"Biometria de {nome} vinculada no SQLite com sucesso!"})


@app.route("/usuarios", methods=["GET"])
def api_listar_usuarios():
    return jsonify(list(banco.listar_usuarios().values()))


@app.route("/usuarios/<int:uid>", methods=["DELETE"])
def api_excluir_usuario(uid: int):
    banco.excluir_usuario(uid)
    motor.treinar_base_completa()
    return jsonify({"ok": True})


@app.route("/logs", methods=["GET"])
def api_listar_logs():
    return jsonify(banco.listar_logs(50))


@app.route("/exportar_csv")
def exportar_csv():
    logs = banco.listar_logs(5000)
    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow(["ID", "DATA_HORA", "USUARIO_ID", "NOME", "NIVEL", "CONFIANCA_%", "RESULTADO", "AQUISICAO"])
    for r in logs:
        writer.writerow([r["id"], r["data_hora"], r["usuario_id"], r["nome"], r["nivel"], r["confianca"], r["resultado"], r["metodo"]])
    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=relatorio_cofre_mma.csv"},
    )


if __name__ == "__main__":
    print("=" * 65)
    print("SISTEMA BIOMÉTRICO DO COFRE — MINISTÉRIO DO MEIO AMBIENTE")
    print("Banco SQLite populado: LUCAS (Tesoureiro - Nível 3), IGOR, GUSTAVO, MOISÉS, GUILHERME (Senha: 1234)")
    print("Acesse no navegador: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)