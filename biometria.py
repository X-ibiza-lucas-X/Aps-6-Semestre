import time
import platform
import cv2
import numpy as np
import face_recognition
from typing import Optional, Tuple, List, Dict


class PipelineBiometrico:
    # Limiar oficial definido no RF-04 do ETP (Distância Euclidiana <= 0.55)
    LIMIAR_DISTANCIA_EUCLIDIANA = 0.55
    # Limiar de Eye Aspect Ratio (EAR) para detecção de piscada ocular (Liveness)
    LIMIAR_EAR_PISCADA = 0.21

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None

    # =========================================================================
    # FASE 1: AQUISIÇÃO (SENSOR) - Captura contínua 640x480 com aquecimento
    # =========================================================================
    def iniciar_camera(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
            self.liberar_camera()

        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        indices = [self.camera_index]
        indices.extend(indice for indice in range(4) if indice != self.camera_index)

        for indice in indices:
            for backend in backends:
                cap_teste = cv2.VideoCapture(indice, backend)
                if not cap_teste.isOpened():
                    cap_teste.release()
                    continue

                cap_teste.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_teste.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap_teste.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ultimo_frame = None
                for _ in range(20):
                    ret, frm = cap_teste.read()
                    if ret and frm is not None and frm.size > 0:
                        ultimo_frame = frm

                if ultimo_frame is not None:
                    self.camera_index = indice
                    self.cap = cap_teste
                    return True
                cap_teste.release()

        return False

    def capturar_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is not None and self.cap.isOpened():
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    frame = cv2.flip(frame, 1)
                    return True, np.ascontiguousarray(frame, dtype=np.uint8)

            self.liberar_camera()

        if self.iniciar_camera():
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    frame = cv2.flip(frame, 1)
                    return True, np.ascontiguousarray(frame, dtype=np.uint8)

        return False, None

    def liberar_camera(self):
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            finally:
                self.cap = None

    # =========================================================================
    # FASE 2: PRÉ-PROCESSAMENTO - Redimensionamento, CLAHE (LAB) e BGR -> RGB
    # =========================================================================
    def preprocessar_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retorna:
        1. frame_rgb_nativo: imagem RGB contígua uint8 na resolução original (640x480)
        2. frame_rgb_reduzido: imagem RGB 0.5x para detecção rápida
        3. frame_bgr_eq: imagem equalizada via CIELAB + CLAHE
        """
        # Conversão limpa BGR -> RGB contígua para o motor dlib/ResNet-34
        frame_rgb_nativo = np.ascontiguousarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )

        # Equalização adaptativa de histograma no canal de luminância (CIELAB + CLAHE)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        frame_bgr_eq = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # Versão reduzida 0.5x para detecção veloz
        frame_reduzido = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
        frame_rgb_reduzido = np.ascontiguousarray(
            cv2.cvtColor(frame_reduzido, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
        return frame_rgb_nativo, frame_rgb_reduzido, frame_bgr_eq

    def avaliar_iluminacao(self, frame_bgr: np.ndarray) -> Tuple[float, str]:
        """Avalia o brilho médio do quadro para orientar o usuário em tempo real."""
        cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brilho_medio = float(np.mean(cinza))
        if brilho_medio < 45:
            return brilho_medio, "Ambiente escuro (melhore a iluminação)"
        if brilho_medio > 225:
            return brilho_medio, "Excesso de luz (evite reflexo direto)"
        return brilho_medio, "Iluminação adequada"

    # =========================================================================
    # FASE 3: SEGMENTAÇÃO - Detecção Facial Adaptativa Multi-Escala (HOG)
    # =========================================================================
    def detectar_rostos_adaptativo(
        self,
        frame_rgb_nativo: np.ndarray,
        frame_rgb_reduzido: np.ndarray,
        frame_bgr_eq: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Tenta detectar o rosto em 3 níveis progressivos para evitar falhas por distância ou luz:
        1. Escala 0.5x (ultra-rápida para rosto próximo)
        2. Escala 1.0x nativa 640x480 (rosto em distância normal de notebook)
        3. Escala 1.0x com equalização CLAHE (ambientes escuros ou contra-luz)
        Retorna coordenadas sempre na escala nativa (640x480) e o frame RGB usado.
        """
        # Tentativa 1: Escala reduzida 0.5x
        locs_red = face_recognition.face_locations(frame_rgb_reduzido, number_of_times_to_upsample=1, model="hog")
        if len(locs_red) > 0:
            locs_nativas = [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in locs_red]
            return locs_nativas, frame_rgb_nativo

        # Tentativa 2: Resolução nativa 1.0x (640x480)
        locs_nat = face_recognition.face_locations(frame_rgb_nativo, number_of_times_to_upsample=1, model="hog")
        if len(locs_nat) > 0:
            return locs_nat, frame_rgb_nativo

        # Tentativa 3: Resolução nativa com equalização CLAHE
        frame_rgb_eq = np.ascontiguousarray(cv2.cvtColor(frame_bgr_eq, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        locs_eq = face_recognition.face_locations(frame_rgb_eq, number_of_times_to_upsample=1, model="hog")
        return locs_eq, frame_rgb_eq

    def rastrear_rosto_rapido(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detecção leve usada no preview ao vivo para desenhar a guia verde em tempo real."""
        frame_rgb_nativo, frame_rgb_reduzido, frame_bgr_eq = self.preprocessar_frame(frame_bgr)
        locs, _ = self.detectar_rostos_adaptativo(frame_rgb_nativo, frame_rgb_reduzido, frame_bgr_eq)
        if len(locs) == 1:
            return locs[0]
        return None

    # =========================================================================
    # FASE 4: EXTRAÇÃO DE CARACTERÍSTICAS - 68 Marcos, EAR Liveness e Embedding 128D
    # =========================================================================
    @staticmethod
    def _calcular_ear_olho(pontos_olho: List[Tuple[int, int]]) -> float:
        """
        Calcula o Eye Aspect Ratio (EAR) usando os 6 marcos anatômicos de cada olho:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        pts = np.array(pontos_olho, dtype=np.float64)
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        if h == 0:
            return 0.30
        return float((v1 + v2) / (2.0 * h))

    def calcular_liveness_ear(self, landmarks: Dict[str, List[Tuple[int, int]]]) -> float:
        """Retorna o EAR médio dos dois olhos (valores < 0.21 indicam piscada)."""
        olho_esq = landmarks.get("left_eye", [])
        olho_dir = landmarks.get("right_eye", [])
        if len(olho_esq) < 6 or len(olho_dir) < 6:
            return 0.30
        ear_esq = self._calcular_ear_olho(olho_esq)
        ear_dir = self._calcular_ear_olho(olho_dir)
        return (ear_esq + ear_dir) / 2.0

    def extrair_caracteristicas_completas(self, frame_bgr: np.ndarray) -> Dict:
        """
        Executa as Fases 2, 3 e 4 do PDI e retorna telemetria completa,
        ROI, 68 marcos faciais, EAR de vivacidade e vetor 128D (ResNet-34).
        """
        t0 = time.perf_counter()
        frame_rgb_nativo, frame_rgb_reduzido, frame_bgr_eq = self.preprocessar_frame(frame_bgr)
        t_prep = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        localizacoes, frame_rgb_usado = self.detectar_rostos_adaptativo(
            frame_rgb_nativo, frame_rgb_reduzido, frame_bgr_eq
        )
        t_seg = (time.perf_counter() - t1) * 1000

        if len(localizacoes) == 0:
            return {
                "sucesso": False,
                "mensagem": "Nenhum rosto detectado. Centralize seu rosto na câmera com boa iluminação.",
                "roi": None,
                "landmarks": None,
                "ear": None,
                "embedding": None,
                "frame_equalizado": frame_bgr_eq,
                "tempos_ms": {"preprocessamento": t_prep, "segmentacao": t_seg, "extracao": 0.0}
            }

        if len(localizacoes) > 1:
            return {
                "sucesso": False,
                "mensagem": "Múltiplos rostos detectados! Apenas o titular deve estar na frente da câmera.",
                "roi": None,
                "landmarks": None,
                "ear": None,
                "embedding": None,
                "frame_equalizado": frame_bgr_eq,
                "tempos_ms": {"preprocessamento": t_prep, "segmentacao": t_seg, "extracao": 0.0}
            }

        t2 = time.perf_counter()
        landmarks_lista = face_recognition.face_landmarks(frame_rgb_usado, face_locations=localizacoes)
        encodings = face_recognition.face_encodings(frame_rgb_usado, known_face_locations=localizacoes, num_jitters=1)
        t_ext = (time.perf_counter() - t2) * 1000

        if not encodings or not landmarks_lista:
            return {
                "sucesso": False,
                "mensagem": "Falha ao mapear os 68 marcos faciais. Mantenha o rosto frontal.",
                "roi": None,
                "landmarks": None,
                "ear": None,
                "embedding": None,
                "frame_equalizado": frame_bgr_eq,
                "tempos_ms": {"preprocessamento": t_prep, "segmentacao": t_seg, "extracao": t_ext}
            }

        roi_original = localizacoes[0]
        landmarks_original = landmarks_lista[0]
        ear_medio = self.calcular_liveness_ear(landmarks_original)

        return {
            "sucesso": True,
            "mensagem": "Características faciais 128D extraídas com sucesso.",
            "roi": roi_original,
            "landmarks": landmarks_original,
            "ear": ear_medio,
            "embedding": np.asarray(encodings[0], dtype=np.float64),
            "frame_equalizado": frame_bgr_eq,
            "tempos_ms": {"preprocessamento": t_prep, "segmentacao": t_seg, "extracao": t_ext}
        }

    @staticmethod
    def consolidar_centroide_cadastro(embeddings: List[np.ndarray]) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Funde múltiplas amostras biométricas (ex: 3 capturas) em um único vetor
        médio de 128 dimensões (centroide), validando a coerência intra-classe.
        """
        if len(embeddings) < 2:
            return False, None, "São necessárias pelo menos 2 amostras para consolidação."

        matriz = np.vstack(embeddings)
        centroide_preliminar = np.mean(matriz, axis=0)

        # Filtra eventuais amostras com movimento brusco (distância > 0.55 do centroide)
        validos = [
            emb for emb in embeddings
            if float(np.linalg.norm(emb - centroide_preliminar)) <= 0.55
        ]
        if len(validos) < 2:
            return False, None, "As amostras capturadas pertencem a rostos distintos. Tente novamente."

        centroide_final = np.mean(np.vstack(validos), axis=0)
        return True, centroide_final, f"Template 128D consolidado com sucesso ({len(validos)} amostras válidas)."

    # =========================================================================
    # FASE 5: CLASSIFICAÇÃO E DECISÃO - Distância Euclidiana <= 0.55 (RF-04)
    # =========================================================================
    def verificar_identidade(
        self,
        frame_bgr: np.ndarray,
        embedding_cadastrado: np.ndarray
    ) -> Dict:
        """
        Executa o pipeline completo (Fases 2 a 5) e valida se a Distância Euclidiana
        é menor ou igual a 0.55 dentro da meta de latência < 2.5s (RNF-01).
        """
        inicio_total = time.perf_counter()

        extracao = self.extrair_caracteristicas_completas(frame_bgr)
        if not extracao["sucesso"]:
            latencia_ms = (time.perf_counter() - inicio_total) * 1000
            return {
                "rosto_detectado": False,
                "aprovado": False,
                "distancia": None,
                "similaridade_cosseno": None,
                "confianca_pct": 0.0,
                "ear": None,
                "latencia_ms": latencia_ms,
                "tempos_ms": extracao["tempos_ms"],
                "roi": None,
                "landmarks": None,
                "mensagem": extracao["mensagem"]
            }

        t_cls_ini = time.perf_counter()
        emb_vivo = extracao["embedding"]

        # 1. Distância Euclidiana (Métrica oficial RF-04)
        distancia = float(np.linalg.norm(embedding_cadastrado - emb_vivo))

        # 2. Similaridade de Cosseno complementar
        norma_prod = np.linalg.norm(embedding_cadastrado) * np.linalg.norm(emb_vivo)
        sim_cosseno = float(np.dot(embedding_cadastrado, emb_vivo) / norma_prod) if norma_prod > 0 else 0.0

        # 3. Decisão binária estrita (<= 0.55)
        aprovado = distancia <= self.LIMIAR_DISTANCIA_EUCLIDIANA
        confianca_pct = max(0.0, min(100.0, (1.0 - (distancia / 1.10)) * 100.0))
        t_cls = (time.perf_counter() - t_cls_ini) * 1000

        latencia_ms = (time.perf_counter() - inicio_total) * 1000
        tempos = extracao["tempos_ms"]
        tempos["classificacao"] = t_cls

        if aprovado:
            msg = f"IDENTIDADE APROVADA (Distância: {distancia:.4f} <= {self.LIMIAR_DISTANCIA_EUCLIDIANA})"
        else:
            msg = f"IDENTIDADE REJEITADA (Distância: {distancia:.4f} > {self.LIMIAR_DISTANCIA_EUCLIDIANA})"

        return {
            "rosto_detectado": True,
            "aprovado": aprovado,
            "distancia": distancia,
            "similaridade_cosseno": sim_cosseno,
            "confianca_pct": confianca_pct,
            "ear": extracao["ear"],
            "latencia_ms": latencia_ms,
            "tempos_ms": tempos,
            "roi": extracao["roi"],
            "landmarks": extracao["landmarks"],
            "mensagem": msg
        }

    # =========================================================================
    # UTILITÁRIO DE VISUALIZAÇÃO (HUD BIOMÉTRICO SOBRE FRAME DE EXIBIÇÃO)
    # =========================================================================
    @staticmethod
    def desenhar_hud_biometrico(
        frame_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
        landmarks: Optional[Dict[str, List[Tuple[int, int]]]],
        cor_bgr: Tuple[int, int, int] = (74, 222, 128)
    ) -> np.ndarray:
        """Desenha bounding box da ROI e os 68 marcos faciais apenas na cópia visual."""
        canvas = frame_bgr.copy()
        if roi is not None:
            top, right, bottom, left = roi
            cv2.rectangle(canvas, (left, top), (right, bottom), cor_bgr, 2)

        if landmarks is not None:
            for regiao, pontos in landmarks.items():
                for (x, y) in pontos:
                    cv2.circle(canvas, (x, y), 2, (56, 189, 248), -1)
        return canvas