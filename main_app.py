import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageTk
from typing import Optional, Dict, List

from database import BancoDados
from biometria import PipelineBiometrico


class SistemaBancarioApp:
    # Limiares de Risco (ETP RF-03 + Níveis de Segurança APS)
    LIMITE_NIVEL_2_BIOMETRIA = 1000.00
    LIMITE_NIVEL_3_COFRE_MAXIMO = 10000.00
    CONTA_COFRE_CENTRAL = "8888-0"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("APS 6° Semestre | UNIP Bank - Validação Biométrica")
        self.root.geometry("1060x710")
        self.root.configure(bg="#0F172A")
        self.root.resizable(False, False)

        self.db = BancoDados()
        self.pipeline = PipelineBiometrico(camera_index=0)
        self.usuario_logado: Optional[Dict] = None

        self._configurar_estilos()
        self.exibir_tela_login()

    def _configurar_estilos(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1E293B",
            foreground="#F8FAFC",
            fieldbackground="#1E293B",
            rowheight=27,
            font=("Arial", 9)
        )
        style.configure(
            "Treeview.Heading",
            background="#334155",
            foreground="#F8FAFC",
            font=("Arial", 9, "bold")
        )
        style.map("Treeview", background=[("selected", "#0284C7")])

    def _limpar_tela(self):
        self.pipeline.liberar_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

    @staticmethod
    def _formatar_brl(valor: float) -> str:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # =========================================================================
    # TELA 1: LOGIN DOS INTEGRANTES DO GRUPO (INC-01 / RF-02)
    # =========================================================================
    def exibir_tela_login(self):
        self._limpar_tela()

        container = tk.Frame(
            self.root, bg="#1E293B", padx=45, pady=35,
            highlightbackground="#334155", highlightthickness=1
        )
        container.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(container, text="UNIP BANK", font=("Arial", 24, "bold"), bg="#1E293B", fg="#38BDF8").pack(pady=(0, 4))
        tk.Label(
            container,
            text="Validação Transacional por Visão Computacional (ResNet-34)",
            font=("Arial", 10),
            bg="#1E293B",
            fg="#94A3B8"
        ).pack(pady=(0, 18))

        dica_equipe = tk.Label(
            container,
            text=(
                "Contas pré-cadastradas (senha: 234):\n"
                "• Gustavo (1001-0)   • Igor (1002-0)\n"
                "• Lucas (1003-0)     • Moisés (1004-0)\n"
                "• Guilherme (1005-0)\n"
                "Após entrar, cadastre sua biometria no botão do painel."
            ),
            font=("Arial", 9),
            bg="#0F172A",
            fg="#38BDF8",
            padx=12,
            pady=8,
            justify="left"
        )
        dica_equipe.pack(fill="x", pady=(0, 18))

        tk.Label(container, text="Conta ou Primeiro Nome (ex: Lucas ou 1003-0):", font=("Arial", 10, "bold"), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_conta = tk.Entry(container, font=("Arial", 12), width=34, bg="#0F172A", fg="#FFFFFF", insertbackground="white")
        entry_conta.pack(pady=(4, 14), ipady=5)
        entry_conta.focus_set()

        tk.Label(container, text="Senha de Acesso:", font=("Arial", 10, "bold"), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_senha = tk.Entry(container, font=("Arial", 12), width=34, show="*", bg="#0F172A", fg="#FFFFFF", insertbackground="white")
        entry_senha.pack(pady=(4, 20), ipady=5)

        def realizar_login(event=None):
            conta = entry_conta.get()
            senha = entry_senha.get()
            if not conta or not senha:
                messagebox.showwarning("Campos Obrigatórios", "Informe a conta (ou primeiro nome) e a senha.")
                return

            sucesso, dados, msg = self.db.autenticar_login(conta, senha)
            if sucesso:
                self.usuario_logado = dados
                self.exibir_painel_bancario()
            else:
                messagebox.showerror("Acesso Negado", msg)

        entry_senha.bind("<Return>", realizar_login)

        tk.Button(
            container, text="ENTRAR NA CONTA", font=("Arial", 11, "bold"),
            bg="#0284C7", fg="white", activebackground="#0369A1", cursor="hand2",
            width=32, pady=9, bd=0, command=realizar_login
        ).pack(pady=(0, 12))

        def iniciar_login_biometrico():
            identificador = entry_conta.get().strip()
            if not identificador:
                messagebox.showwarning(
                    "Conta Obrigatória",
                    "Informe seu nome ou número de conta para validar a biometria."
                )
                return

            pronto, usuario, mensagem = self.db.obter_usuario_login_biometrico(identificador)
            if not pronto or usuario is None:
                messagebox.showerror("Login Facial Indisponível", mensagem)
                return

            if not self.pipeline.iniciar_camera():
                messagebox.showerror("Erro de Câmera", "Não foi possível iniciar a webcam.")
                return

            modal = tk.Toplevel(self.root)
            modal.title("Entrar com Biometria Facial")
            modal.geometry("520x500")
            modal.configure(bg="#1E293B")
            modal.transient(self.root)
            modal.grab_set()

            tk.Label(
                modal,
                text=f"LOGIN FACIAL - {usuario['nome']}",
                font=("Arial", 12, "bold"),
                bg="#0284C7",
                fg="white",
                pady=10
            ).pack(fill="x")

            lbl_camera = tk.Label(modal, bg="#000000", width=420, height=315)
            lbl_camera.pack(pady=12)

            lbl_status = tk.Label(
                modal,
                text="Centralize seu rosto e clique em Validar identidade.",
                font=("Arial", 10, "bold"),
                bg="#0F172A",
                fg="#FBBF24",
                wraplength=460,
                pady=10,
                padx=12
            )
            lbl_status.pack(fill="x", padx=25, pady=(0, 10))

            estado = {"ativo": True, "frame": None}

            def fechar_modal():
                estado["ativo"] = False
                self.pipeline.liberar_camera()
                modal.destroy()

            modal.protocol("WM_DELETE_WINDOW", fechar_modal)

            def atualizar_preview():
                if not estado["ativo"]:
                    return
                ok, frame = self.pipeline.capturar_frame()
                if ok and frame is not None:
                    estado["frame"] = frame
                    exibicao = cv2.resize(frame, (420, 315))
                    imagem = ImageTk.PhotoImage(
                        image=Image.fromarray(cv2.cvtColor(exibicao, cv2.COLOR_BGR2RGB))
                    )
                    lbl_camera.imgtk = imagem
                    lbl_camera.configure(image=imagem)
                else:
                    lbl_status.configure(
                        text="Sem imagem da webcam. Verifique a câmera e tente novamente.",
                        fg="#F87171"
                    )
                modal.after(35, atualizar_preview)

            def validar_rosto():
                frame = estado["frame"]
                if frame is None:
                    lbl_status.configure(text="Aguardando imagem da webcam...", fg="#FBBF24")
                    return

                botao_validar.configure(state="disabled")
                lbl_status.configure(text="Analisando o rosto...", fg="#38BDF8")
                modal.update_idletasks()

                try:
                    extracao = self.pipeline.extrair_caracteristicas_completas(frame.copy())
                except (RuntimeError, ValueError) as erro:
                    lbl_status.configure(
                        text=f"Não foi possível analisar a imagem: {erro}",
                        fg="#F87171"
                    )
                    botao_validar.configure(state="normal")
                    return

                if not extracao["sucesso"]:
                    lbl_status.configure(text=extracao["mensagem"], fg="#F87171")
                    botao_validar.configure(state="normal")
                    return

                sucesso, dados, mensagem = self.db.autenticar_login_biometrico(
                    usuario["numero_conta"],
                    extracao["embedding"]
                )
                if not sucesso or dados is None:
                    lbl_status.configure(text=mensagem, fg="#F87171")
                    botao_validar.configure(state="normal")
                    return

                estado["ativo"] = False
                self.pipeline.liberar_camera()
                modal.destroy()
                self.usuario_logado = dados
                self.exibir_painel_bancario()

            botoes = tk.Frame(modal, bg="#1E293B")
            botoes.pack(pady=(0, 12))
            botao_validar = tk.Button(
                botoes,
                text="VALIDAR IDENTIDADE",
                font=("Arial", 10, "bold"),
                bg="#16A34A",
                fg="white",
                padx=18,
                pady=8,
                bd=0,
                cursor="hand2",
                command=validar_rosto
            )
            botao_validar.pack(side="left", padx=5)

            tk.Button(
                botoes,
                text="Voltar",
                font=("Arial", 10),
                bg="#475569",
                fg="white",
                padx=18,
                pady=8,
                bd=0,
                cursor="hand2",
                command=fechar_modal
            ).pack(side="left", padx=5)

            atualizar_preview()

        tk.Button(
            container, text="ENTRAR COM BIOMETRIA FACIAL", font=("Arial", 10, "bold"),
            bg="#059669", fg="white", activebackground="#047857", cursor="hand2",
            width=32, pady=8, bd=0, command=iniciar_login_biometrico
        ).pack(pady=(0, 12))

        tk.Button(
            container, text="Criar Nova Conta Extra + Cadastro Biométrico", font=("Arial", 10),
            bg="#334155", fg="#E2E8F0", activebackground="#475569", cursor="hand2",
            width=36, pady=7, bd=0, command=self.exibir_tela_cadastro
        ).pack()

    # =========================================================================
    # TELA 2: CADASTRO DE NOVA CONTA EXTRA (RNF-03 / LGPD)
    # =========================================================================
    def exibir_tela_cadastro(self):
        self._limpar_tela()

        frame_principal = tk.Frame(self.root, bg="#0F172A")
        frame_principal.pack(fill="both", expand=True, padx=25, pady=20)

        col_esq = tk.Frame(frame_principal, bg="#1E293B", padx=25, pady=20)
        col_esq.pack(side="left", fill="both", expand=True, padx=(0, 15))

        tk.Label(col_esq, text="Abertura de Conta Digital", font=("Arial", 17, "bold"), bg="#1E293B", fg="#38BDF8").pack(anchor="w", pady=(0, 12))

        tk.Label(col_esq, text="Nome Completo do Titular:", font=("Arial", 10), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_nome = tk.Entry(col_esq, font=("Arial", 11), bg="#0F172A", fg="white", insertbackground="white")
        entry_nome.pack(fill="x", pady=(3, 10), ipady=4)

        tk.Label(col_esq, text="Número da Conta (ex: 1020-5):", font=("Arial", 10), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_conta = tk.Entry(col_esq, font=("Arial", 11), bg="#0F172A", fg="white", insertbackground="white")
        entry_conta.pack(fill="x", pady=(3, 10), ipady=4)

        tk.Label(col_esq, text="Senha de Acesso (mín. 4 caracteres):", font=("Arial", 10), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_senha = tk.Entry(col_esq, font=("Arial", 11), show="*", bg="#0F172A", fg="white", insertbackground="white")
        entry_senha.pack(fill="x", pady=(3, 10), ipady=4)

        tk.Label(col_esq, text="Saldo Inicial Simulado (R$):", font=("Arial", 10), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_saldo = tk.Entry(col_esq, font=("Arial", 11), bg="#0F172A", fg="white", insertbackground="white")
        entry_saldo.insert(0, "25000.00")
        entry_saldo.pack(fill="x", pady=(3, 14), ipady=4)

        lbl_progresso = tk.Label(
            col_esq,
            text="Amostras Biométricas Coletadas: 0 / 3",
            font=("Arial", 10, "bold"),
            bg="#1E293B",
            fg="#38BDF8"
        )
        lbl_progresso.pack(anchor="w", pady=(0, 6))

        lbl_status_bio = tk.Label(
            col_esq,
            text="Aguarde o quadro verde no rosto e clique em 'Coletar 3 Amostras (Auto)' ou 'Capturar 1 Amostra'.",
            font=("Arial", 9),
            bg="#0F172A",
            fg="#FBBF24",
            padx=10,
            pady=10,
            wraplength=360,
            justify="left"
        )
        lbl_status_bio.pack(fill="x", pady=(0, 15))

        amostras_embeddings: List[np.ndarray] = []
        vetor_consolidado = {"embedding": None}

        col_dir = tk.Frame(frame_principal, bg="#1E293B", padx=20, pady=20)
        col_dir.pack(side="right", fill="both", expand=True)

        tk.Label(col_dir, text="Sensor Óptico 640x480 (Privacidade por Design - RNF-03)", font=("Arial", 11, "bold"), bg="#1E293B", fg="#E2E8F0").pack(pady=(0, 4))
        lbl_luz = tk.Label(col_dir, text="Inicializando câmera...", font=("Arial", 9, "bold"), bg="#1E293B", fg="#94A3B8")
        lbl_luz.pack(pady=(0, 6))

        lbl_video = tk.Label(col_dir, bg="#000000", width=460, height=340)
        lbl_video.pack(pady=4)

        estado_cam = {
            "rodando": True,
            "ultimo_frame": None,
            "roi_atual": None,
            "ticks": 0,
            "auto_coletando": False
        }

        cam_ok = self.pipeline.iniciar_camera()
        if not cam_ok:
            lbl_luz.configure(text="Webcam não detectada. Use 'Carregar Foto (Arquivo)' abaixo.", fg="#F87171")

        def atualizar_preview():
            if not estado_cam["rodando"]:
                return
            if cam_ok:
                ok, frame = self.pipeline.capturar_frame()
                if ok and frame is not None:
                    estado_cam["ultimo_frame"] = frame
                    estado_cam["ticks"] += 1

                    # Rastreia rosto ao vivo a cada 6 frames para dar retorno visual imediato
                    if estado_cam["ticks"] % 6 == 0:
                        estado_cam["roi_atual"] = self.pipeline.rastrear_rosto_rapido(frame)
                        _, status_luz = self.pipeline.avaliar_iluminacao(frame)
                        if estado_cam["roi_atual"] is not None:
                            lbl_luz.configure(text=f"ROSTO DETECTADO (Pronto) | {status_luz}", fg="#4ADE80")
                        else:
                            lbl_luz.configure(text=f"Aguardando rosto no quadro | {status_luz}", fg="#FBBF24")

                    frame_vis = frame.copy()
                    if estado_cam["roi_atual"] is not None:
                        t, r, b, l = estado_cam["roi_atual"]
                        cv2.rectangle(frame_vis, (l, t), (r, b), (74, 222, 128), 3)
                    else:
                        cv2.rectangle(frame_vis, (180, 70), (460, 410), (56, 189, 248), 2)

                    frame_exibicao = cv2.resize(frame_vis, (460, 340))
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2RGB)))
                    lbl_video.imgtk = img_tk
                    lbl_video.configure(image=img_tk)
            self.root.after(25, atualizar_preview)

        atualizar_preview()

        def processar_frame_amostra(frame_bgr: np.ndarray) -> bool:
            if len(amostras_embeddings) >= 3:
                return True

            lbl_status_bio.configure(text="Extraindo 68 marcos faciais e vetor 128D (ResNet-34)...", fg="#38BDF8")
            self.root.update_idletasks()

            resultado = self.pipeline.extrair_caracteristicas_completas(frame_bgr)
            if not resultado["sucesso"]:
                lbl_status_bio.configure(text=resultado["mensagem"], fg="#F87171")
                return False

            frame_hud = self.pipeline.desenhar_hud_biometrico(
                frame_bgr, resultado["roi"], resultado["landmarks"]
            )
            exibicao = cv2.resize(frame_hud, (460, 340))
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(exibicao, cv2.COLOR_BGR2RGB)))
            lbl_video.imgtk = img_tk
            lbl_video.configure(image=img_tk)

            amostras_embeddings.append(resultado["embedding"])
            qtd = len(amostras_embeddings)
            lbl_progresso.configure(text=f"Amostras Biométricas Coletadas: {qtd} / 3")

            if qtd < 3:
                lbl_status_bio.configure(
                    text=f"Amostra {qtd}/3 coletada com sucesso! Capturando próxima...",
                    fg="#4ADE80"
                )
            else:
                ok_cons, centroide, msg_cons = self.pipeline.consolidar_centroide_cadastro(amostras_embeddings)
                if ok_cons:
                    vetor_consolidado["embedding"] = centroide
                    lbl_status_bio.configure(
                        text=f"{msg_cons} Pronto para concluir o cadastro!",
                        fg="#4ADE80"
                    )
                else:
                    amostras_embeddings.clear()
                    lbl_progresso.configure(text="Amostras Biométricas Coletadas: 0 / 3")
                    lbl_status_bio.configure(text=msg_cons, fg="#F87171")
                    return False
            return True

        def capturar_manual():
            if estado_cam["ultimo_frame"] is None:
                messagebox.showwarning("Sem Quadro", "Aguarde a imagem da câmera estabilizar ou carregue uma foto.")
                return
            processar_frame_amostra(estado_cam["ultimo_frame"].copy())

        def iniciar_coleta_automatica():
            if estado_cam["ultimo_frame"] is None:
                messagebox.showwarning("Sem Quadro", "Webcam indisponível. Use o botão 'Carregar Foto'.")
                return
            if len(amostras_embeddings) >= 3:
                messagebox.showinfo("Concluído", "As 3 amostras já foram coletadas!")
                return
            estado_cam["auto_coletando"] = True

            def passo_auto():
                if not estado_cam["rodando"] or not estado_cam["auto_coletando"]:
                    return
                if len(amostras_embeddings) >= 3:
                    estado_cam["auto_coletando"] = False
                    return
                if estado_cam["ultimo_frame"] is not None:
                    processar_frame_amostra(estado_cam["ultimo_frame"].copy())
                if len(amostras_embeddings) < 3:
                    self.root.after(650, passo_auto)
                else:
                    estado_cam["auto_coletando"] = False

            passo_auto()

        def carregar_foto_arquivo():
            caminho = filedialog.askopenfilename(
                title="Selecione uma foto frontal nítida",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not caminho:
                return
            img = cv2.imread(caminho)
            if img is None:
                messagebox.showerror("Erro", "Não foi possível ler a imagem selecionada.")
                return
            img = cv2.resize(img, (640, 480))
            amostras_embeddings.clear()
            for _ in range(3):
                if not processar_frame_amostra(img):
                    break

        def finalizar_cadastro():
            nome = entry_nome.get().strip()
            conta = entry_conta.get().strip()
            senha = entry_senha.get().strip()

            if not nome or not conta or not senha:
                messagebox.showwarning("Dados Incompletos", "Preencha nome, conta e senha.")
                return
            if vetor_consolidado["embedding"] is None:
                messagebox.showwarning("Biometria Obrigatória", "Colete as 3 amostras faciais antes de concluir.")
                return

            try:
                saldo = float(entry_saldo.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Saldo Inválido", "Informe um valor numérico válido.")
                return

            sucesso, msg = self.db.cadastrar_usuario(nome, conta, senha, vetor_consolidado["embedding"], saldo)
            if sucesso:
                estado_cam["rodando"] = False
                self.pipeline.liberar_camera()
                messagebox.showinfo("Cadastro Concluído", msg)
                self.exibir_tela_login()
            else:
                messagebox.showerror("Erro no Cadastro", msg)

        botoes_cam = tk.Frame(col_dir, bg="#1E293B")
        botoes_cam.pack(pady=(10, 0))

        tk.Button(
            botoes_cam, text="COLETAR 3 AMOSTRAS (AUTO)", font=("Arial", 9, "bold"),
            bg="#16A34A", fg="white", pady=8, padx=10, bd=0, cursor="hand2",
            command=iniciar_coleta_automatica
        ).pack(side="left", padx=4)

        tk.Button(
            botoes_cam, text="Capturar 1 Amostra", font=("Arial", 9, "bold"),
            bg="#0EA5E9", fg="white", pady=8, padx=10, bd=0, cursor="hand2",
            command=capturar_manual
        ).pack(side="left", padx=4)

        tk.Button(
            botoes_cam, text="Carregar Foto", font=("Arial", 9),
            bg="#475569", fg="white", pady=8, padx=10, bd=0, cursor="hand2",
            command=carregar_foto_arquivo
        ).pack(side="left", padx=4)

        tk.Button(
            col_esq, text="CONCLUIR CADASTRO", font=("Arial", 11, "bold"),
            bg="#16A34A", fg="white", pady=9, bd=0, cursor="hand2",
            command=finalizar_cadastro
        ).pack(fill="x", pady=(0, 8))

        tk.Button(
            col_esq, text="Voltar ao Login", font=("Arial", 10),
            bg="#334155", fg="white", pady=7, bd=0, cursor="hand2",
            command=lambda: [estado_cam.update({"rodando": False}), self.exibir_tela_login()]
        ).pack(fill="x")

    # =========================================================================
    # MODAL: VINCULAR / ATUALIZAR BIOMETRIA DO USUÁRIO PRÉ-CADASTRADO
    # =========================================================================
    def abrir_modal_vincular_biometria(self, callback_apos_cadastro=None):
        modal = tk.Toplevel(self.root)
        modal.title("Cadastro de Template Biométrico 128D (LGPD)")
        modal.geometry("600x600")
        modal.configure(bg="#1E293B")
        modal.transient(self.root)
        modal.grab_set()

        tk.Label(
            modal,
            text=f"VÍNCULO BIOMÉTRICO - {self.usuario_logado['nome']}",
            font=("Arial", 11, "bold"),
            bg="#0284C7",
            fg="#FFFFFF",
            pady=10
        ).pack(fill="x")

        lbl_progresso = tk.Label(
            modal,
            text="Amostras Coletadas: 0 / 3  (Aguarde o retângulo verde no rosto)",
            font=("Arial", 10, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
            pady=6
        )
        lbl_progresso.pack()

        lbl_rastreio = tk.Label(
            modal,
            text="Inicializando sensor óptico...",
            font=("Arial", 9, "bold"),
            bg="#1E293B",
            fg="#94A3B8"
        )
        lbl_rastreio.pack(pady=(0, 4))

        lbl_cam = tk.Label(modal, bg="#000000", width=460, height=320)
        lbl_cam.pack(pady=4)

        lbl_status = tk.Label(
            modal,
            text="Clique em 'Coletar 3 Amostras (Auto)' ou 'Capturar 1 Amostra' para registrar seu vetor 128D.",
            font=("Arial", 9, "bold"),
            bg="#0F172A",
            fg="#FBBF24",
            pady=10,
            padx=12,
            wraplength=520
        )
        lbl_status.pack(fill="x", padx=30, pady=8)

        cam_ok = self.pipeline.iniciar_camera()
        if not cam_ok:
            lbl_rastreio.configure(text="Webcam indisponível. Use 'Carregar Foto (Arquivo)' abaixo.", fg="#F87171")

        estado = {
            "ativo": True,
            "frame_atual": None,
            "roi_atual": None,
            "ticks": 0,
            "auto_coletando": False
        }
        amostras: List[np.ndarray] = []

        def fechar():
            estado["ativo"] = False
            estado["auto_coletando"] = False
            self.pipeline.liberar_camera()
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", fechar)

        def loop_preview():
            if not estado["ativo"]:
                return
            if cam_ok:
                ok, frame = self.pipeline.capturar_frame()
                if ok and frame is not None:
                    estado["frame_atual"] = frame
                    estado["ticks"] += 1

                    if estado["ticks"] % 6 == 0:
                        estado["roi_atual"] = self.pipeline.rastrear_rosto_rapido(frame)
                        if estado["roi_atual"] is not None:
                            lbl_rastreio.configure(text="ROSTO DETECTADO (Pronto para capturar)", fg="#4ADE80")
                        else:
                            lbl_rastreio.configure(text="Posicione seu rosto frente à câmera...", fg="#FBBF24")

                    frame_vis = frame.copy()
                    if estado["roi_atual"] is not None:
                        t, r, b, l = estado["roi_atual"]
                        cv2.rectangle(frame_vis, (l, t), (r, b), (74, 222, 128), 3)
                    else:
                        cv2.rectangle(frame_vis, (180, 70), (460, 410), (56, 189, 248), 2)

                    exib = cv2.resize(frame_vis, (460, 320))
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(exib, cv2.COLOR_BGR2RGB)))
                    lbl_cam.imgtk = img_tk
                    lbl_cam.configure(image=img_tk)
            modal.after(25, loop_preview)

        def processar_amostra_modal(frame_bgr: np.ndarray) -> bool:
            lbl_status.configure(text="Extraindo 68 marcos faciais e embedding 128D (ResNet-34)...", fg="#38BDF8")
            modal.update_idletasks()

            res = self.pipeline.extrair_caracteristicas_completas(frame_bgr)
            if not res["sucesso"]:
                lbl_status.configure(text=res["mensagem"], fg="#F87171")
                return False

            frame_hud = self.pipeline.desenhar_hud_biometrico(
                frame_bgr, res["roi"], res["landmarks"]
            )
            exib = cv2.resize(frame_hud, (460, 320))
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(exib, cv2.COLOR_BGR2RGB)))
            lbl_cam.imgtk = img_tk
            lbl_cam.configure(image=img_tk)

            amostras.append(res["embedding"])
            qtd = len(amostras)
            lbl_progresso.configure(text=f"Amostras Coletadas: {qtd} / 3")

            if qtd < 3:
                lbl_status.configure(
                    text=f"Amostra {qtd}/3 registrada! Mova levemente o ângulo do rosto...",
                    fg="#4ADE80"
                )
                return True

            # Consolidando o centroide após 3 amostras
            ok_cons, centroide, msg_cons = self.pipeline.consolidar_centroide_cadastro(amostras)
            if not ok_cons:
                amostras.clear()
                lbl_progresso.configure(text="Amostras Coletadas: 0 / 3")
                lbl_status.configure(text=msg_cons, fg="#F87171")
                return False

            ok_db, msg_db = self.db.atualizar_biometria_usuario(
                self.usuario_logado["numero_conta"], centroide
            )
            fechar()
            if ok_db:
                self.usuario_logado = self.db.obter_dados_atualizados(self.usuario_logado["numero_conta"])
                messagebox.showinfo(
                    "Biometria Ativa",
                    f"{msg_db}\n{msg_cons}\nNenhuma foto bruta foi salva no banco (LGPD)."
                )
                self.exibir_painel_bancario()
                if callback_apos_cadastro:
                    callback_apos_cadastro()
            else:
                messagebox.showerror("Erro", msg_db)
            return True

        def capturar_uma():
            if estado["frame_atual"] is None:
                messagebox.showwarning("Sem Quadro", "Webcam indisponível. Use 'Carregar Foto'.", parent=modal)
                return
            processar_amostra_modal(estado["frame_atual"].copy())

        def coletar_tres_auto():
            if estado["frame_atual"] is None:
                messagebox.showwarning("Sem Quadro", "Webcam indisponível. Use 'Carregar Foto'.", parent=modal)
                return
            estado["auto_coletando"] = True

            def passo():
                if not estado["ativo"] or not estado["auto_coletando"]:
                    return
                if len(amostras) >= 3:
                    estado["auto_coletando"] = False
                    return
                if estado["frame_atual"] is not None:
                    processar_amostra_modal(estado["frame_atual"].copy())
                if estado["ativo"] and len(amostras) < 3:
                    modal.after(650, passo)
                else:
                    estado["auto_coletando"] = False

            passo()

        def carregar_foto_modal():
            caminho = filedialog.askopenfilename(
                parent=modal,
                title="Selecione uma foto de referência nítida",
                filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
            )
            if not caminho:
                return
            img = cv2.imread(caminho)
            if img is None:
                messagebox.showerror("Erro", "Falha ao abrir imagem.", parent=modal)
                return
            img = cv2.resize(img, (640, 480))
            amostras.clear()
            for _ in range(3):
                if not processar_amostra_modal(img):
                    break

        frame_btns = tk.Frame(modal, bg="#1E293B")
        frame_btns.pack(pady=(2, 8))

        tk.Button(
            frame_btns, text="COLETAR 3 AMOSTRAS (AUTO)", font=("Arial", 9, "bold"),
            bg="#16A34A", fg="white", pady=9, padx=12, bd=0, cursor="hand2",
            command=coletar_tres_auto
        ).pack(side="left", padx=4)

        tk.Button(
            frame_btns, text="Capturar 1 Amostra", font=("Arial", 9, "bold"),
            bg="#0EA5E9", fg="white", pady=9, padx=12, bd=0, cursor="hand2",
            command=capturar_uma
        ).pack(side="left", padx=4)

        tk.Button(
            frame_btns, text="Carregar Foto", font=("Arial", 9),
            bg="#475569", fg="white", pady=9, padx=12, bd=0, cursor="hand2",
            command=carregar_foto_modal
        ).pack(side="left", padx=4)

        loop_preview()

    # =========================================================================
    # TELA 3: PAINEL BANCÁRIO, 3 NÍVEIS DE RISCO, COFRE E EXTRATO AUDITÁVEL
    # =========================================================================
    def exibir_painel_bancario(self):
        self._limpar_tela()
        self.usuario_logado = self.db.obter_dados_atualizados(self.usuario_logado["numero_conta"])
        bio_ativa = self.usuario_logado["biometria_cadastrada"]

        header = tk.Frame(self.root, bg="#1E293B", height=65, padx=22)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text=f"UNIP BANK  |  {self.usuario_logado['nome']}  (Conta: {self.usuario_logado['numero_conta']})",
            font=("Arial", 11, "bold"),
            bg="#1E293B",
            fg="#F8FAFC"
        ).pack(side="left", pady=18)

        tk.Button(
            header, text="Sair", font=("Arial", 9, "bold"),
            bg="#DC2626", fg="white", padx=12, pady=4, bd=0, cursor="hand2",
            command=self.exibir_tela_login
        ).pack(side="right", pady=16, padx=(6, 0))

        tk.Button(
            header, text="Inspetor PDI (5 Fases)", font=("Arial", 9, "bold"),
            bg="#7C3AED", fg="white", padx=12, pady=4, bd=0, cursor="hand2",
            command=self.abrir_inspetor_pdi_5_fases
        ).pack(side="right", pady=16, padx=(6, 0))

        cor_btn_bio = "#059669" if bio_ativa else "#D97706"
        texto_btn_bio = "Atualizar Biometria (128D)" if bio_ativa else "Cadastrar Minha Biometria (3 Amostras)"
        tk.Button(
            header, text=texto_btn_bio, font=("Arial", 9, "bold"),
            bg=cor_btn_bio, fg="white", padx=12, pady=4, bd=0, cursor="hand2",
            command=self.abrir_modal_vincular_biometria
        ).pack(side="right", pady=16)

        corpo = tk.Frame(self.root, bg="#0F172A", padx=25, pady=18)
        corpo.pack(fill="both", expand=True)

        topo_cards = tk.Frame(corpo, bg="#0F172A")
        topo_cards.pack(fill="x", pady=(0, 18))

        card_saldo = tk.Frame(topo_cards, bg="#1E293B", padx=22, pady=16, width=375, height=255)
        card_saldo.pack(side="left", fill="y", padx=(0, 15))
        card_saldo.pack_propagate(False)

        tk.Label(card_saldo, text="SALDO DISPONÍVEL", font=("Arial", 10, "bold"), bg="#1E293B", fg="#94A3B8").pack(anchor="w")
        tk.Label(
            card_saldo,
            text=self._formatar_brl(self.usuario_logado["saldo"]),
            font=("Arial", 21, "bold"),
            bg="#1E293B",
            fg="#4ADE80"
        ).pack(anchor="w", pady=(4, 6))

        status_bio_txt = "Status Facial: ATIVA (Vetor 128D Vinculado)" if bio_ativa else "Status Facial: PENDENTE (Cadastre no botão acima)"
        status_bio_cor = "#4ADE80" if bio_ativa else "#FBBF24"
        tk.Label(card_saldo, text=status_bio_txt, font=("Arial", 9, "bold"), bg="#1E293B", fg=status_bio_cor).pack(anchor="w", pady=(0, 8))

        tk.Label(
            card_saldo,
            text=(
                "Níveis de Segurança (ETP + Cofre APS):\n"
                "• Nível 1 (< R$ 1.000): Liberação imediata\n"
                "• Nível 2 (>= R$ 1.000): Biometria Facial 1:1\n"
                "• Nível 3 (>= R$ 10k ou Cofre 8888-0): Facial + Piscada + Senha"
            ),
            font=("Arial", 8),
            bg="#0F172A",
            fg="#38BDF8",
            padx=10,
            pady=8,
            justify="left"
        ).pack(fill="x")

        card_transf = tk.Frame(topo_cards, bg="#1E293B", padx=22, pady=16, height=255)
        card_transf.pack(side="right", fill="both", expand=True)
        card_transf.pack_propagate(False)

        tk.Label(card_transf, text="Transferência Bancária & Operações de Cofre", font=("Arial", 13, "bold"), bg="#1E293B", fg="#F8FAFC").pack(anchor="w", pady=(0, 10))

        linha_inputs = tk.Frame(card_transf, bg="#1E293B")
        linha_inputs.pack(fill="x", pady=(0, 10))

        frame_dest = tk.Frame(linha_inputs, bg="#1E293B")
        frame_dest.pack(side="left", fill="x", expand=True, padx=(0, 15))
        tk.Label(frame_dest, text="Chave Pix (nome do usuário ou número da conta):", font=("Arial", 10), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_destino = tk.Entry(frame_dest, font=("Arial", 12), bg="#0F172A", fg="white", insertbackground="white")
        entry_destino.insert(0, "8888-0")
        entry_destino.pack(fill="x", pady=(4, 0), ipady=5)

        frame_val = tk.Frame(linha_inputs, bg="#1E293B")
        frame_val.pack(side="right", fill="x", expand=True)
        tk.Label(frame_val, text="Valor da Operação (R$):", font=("Arial", 10), bg="#1E293B", fg="#E2E8F0").pack(anchor="w")
        entry_valor = tk.Entry(frame_val, font=("Arial", 12), bg="#0F172A", fg="white", insertbackground="white")
        entry_valor.pack(fill="x", pady=(4, 0), ipady=5)

        lbl_indicador_risco = tk.Label(
            card_transf,
            text="Classificação de Risco: Digite um valor para visualizar o nível exigido.",
            font=("Arial", 9, "bold"),
            bg="#1E293B",
            fg="#94A3B8"
        )
        lbl_indicador_risco.pack(anchor="w", pady=(2, 10))

        def atualizar_classificacao_risco(*args):
            texto = entry_valor.get().replace(",", ".").strip()
            dest = entry_destino.get().strip()
            destino_encontrado, conta_destino, _ = self.db.resolver_destinatario(dest)
            destino_cofre = destino_encontrado and conta_destino == self.CONTA_COFRE_CENTRAL
            try:
                v = float(texto)
                if v <= 0:
                    lbl_indicador_risco.configure(text="Valor inválido.", fg="#F87171")
                elif v >= self.LIMITE_NIVEL_3_COFRE_MAXIMO or destino_cofre:
                    lbl_indicador_risco.configure(
                        text="NÍVEL 3 (COFRE MÁXIMO): Exige Facial 1:1 + Piscada Ocular (EAR) + Senha.",
                        fg="#F87171"
                    )
                elif v >= self.LIMITE_NIVEL_2_BIOMETRIA:
                    lbl_indicador_risco.configure(
                        text="NÍVEL 2 (ALTO VALOR >= R$ 1.000): Exige Validação Facial 1:1 (Distância <= 0.55).",
                        fg="#FBBF24"
                    )
                else:
                    lbl_indicador_risco.configure(
                        text="NÍVEL 1 (BAIXO RISCO < R$ 1.000): Liberação imediata sem fricção biométrica.",
                        fg="#4ADE80"
                    )
            except ValueError:
                lbl_indicador_risco.configure(
                    text="Classificação de Risco: Digite um valor para visualizar o nível exigido.",
                    fg="#94A3B8"
                )

        entry_valor.bind("<KeyRelease>", atualizar_classificacao_risco)
        entry_destino.bind("<KeyRelease>", atualizar_classificacao_risco)

        def iniciar_fluxo_transferencia():
            destino = entry_destino.get().strip()
            try:
                valor = float(entry_valor.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Valor Inválido", "Digite um valor numérico válido.")
                return

            if valor <= 0:
                messagebox.showwarning("Valor Inválido", "O valor deve ser maior que zero.")
                return
            if valor > self.usuario_logado["saldo"]:
                messagebox.showerror("Saldo Insuficiente", "Saldo insuficiente para esta operação.")
                return

            existe, conta_destino, nome_dest = self.db.resolver_destinatario(destino)
            if not existe or conta_destino is None or conta_destino == self.usuario_logado["numero_conta"]:
                messagebox.showerror(
                    "Destinatário Inválido",
                    "Não foi possível localizar um destinatário único por esse nome/conta, "
                    "ou a chave Pix corresponde à sua própria conta."
                )
                return

            exige_nivel_3 = (valor >= self.LIMITE_NIVEL_3_COFRE_MAXIMO) or (conta_destino == self.CONTA_COFRE_CENTRAL)
            exige_nivel_2 = (valor >= self.LIMITE_NIVEL_2_BIOMETRIA)

            if not exige_nivel_2 and not exige_nivel_3:
                sucesso, msg = self.db.executar_transferencia(
                    conta_origem=self.usuario_logado["numero_conta"],
                    conta_destino=conta_destino,
                    valor=valor,
                    nivel_seguranca="Nível 1 (< R$ 1k)",
                    exigiu_biometria=False
                )
                if sucesso:
                    messagebox.showinfo("Transferência Aprovada", f"{msg}\nDestinatário: {nome_dest}\nModalidade: Nível 1")
                    self.exibir_painel_bancario()
                else:
                    messagebox.showerror("Erro", msg)
                return

            nivel_alvo = 3 if exige_nivel_3 else 2
            if not self.usuario_logado["biometria_cadastrada"]:
                quer_cadastrar = messagebox.askyesno(
                    "Biometria Não Cadastrada",
                    "Esta operação exige validação facial, mas sua conta ainda não possui template biométrico 128D.\n\nDeseja capturar suas 3 amostras faciais agora?"
                )
                if quer_cadastrar:
                    self.abrir_modal_vincular_biometria(
                        callback_apos_cadastro=lambda: self.abrir_modal_validacao_facial(conta_destino, nome_dest, valor, nivel=nivel_alvo)
                    )
                return

            self.abrir_modal_validacao_facial(conta_destino, nome_dest, valor, nivel=nivel_alvo)

        tk.Button(
            card_transf, text="EXECUTAR TRANSFERÊNCIA / ACESSAR COFRE", font=("Arial", 11, "bold"),
            bg="#0284C7", fg="white", pady=9, bd=0, cursor="hand2",
            command=iniciar_fluxo_transferencia
        ).pack(fill="x")

        barra_extrato = tk.Frame(corpo, bg="#0F172A")
        barra_extrato.pack(fill="x", pady=(0, 8))

        tk.Label(barra_extrato, text="Histórico de Auditoria Transacional & Biométrica", font=("Arial", 12, "bold"), bg="#0F172A", fg="#E2E8F0").pack(side="left")

        filtro_var = tk.StringVar(value="TODOS")

        def exportar_csv():
            caminho = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Arquivo CSV", "*.csv")],
                initialfile=f"auditoria_biometrica_{self.usuario_logado['numero_conta']}.csv"
            )
            if caminho:
                ok, msg = self.db.exportar_relatorio_auditoria_csv(self.usuario_logado["numero_conta"], caminho)
                if ok:
                    messagebox.showinfo("Relatório Exportado", msg)
                else:
                    messagebox.showerror("Erro", msg)

        tk.Button(
            barra_extrato, text="Exportar Relatório CSV", font=("Arial", 9, "bold"),
            bg="#059669", fg="white", padx=12, pady=4, bd=0, cursor="hand2",
            command=exportar_csv
        ).pack(side="right", padx=(10, 0))

        colunas = ("data", "destino", "valor", "nivel", "distancia", "latencia", "status")
        tabela = ttk.Treeview(corpo, columns=colunas, show="headings", height=9)
        tabela.heading("data", text="Data/Hora")
        tabela.heading("destino", text="Destino")
        tabela.heading("valor", text="Valor")
        tabela.heading("nivel", text="Nível / Bio")
        tabela.heading("distancia", text="Dist. Euclidiana")
        tabela.heading("latencia", text="Latência")
        tabela.heading("status", text="Status da Operação")

        tabela.column("data", width=130, anchor="center")
        tabela.column("destino", width=85, anchor="center")
        tabela.column("valor", width=105, anchor="e")
        tabela.column("nivel", width=140, anchor="center")
        tabela.column("distancia", width=110, anchor="center")
        tabela.column("latencia", width=80, anchor="center")
        tabela.column("status", width=270, anchor="center")

        def carregar_tabela():
            for row in tabela.get_children():
                tabela.delete(row)
            for item in self.db.listar_extrato(self.usuario_logado["numero_conta"], filtro_var.get()):
                dist_str = f"{item['distancia_euclidiana']:.4f}" if item["distancia_euclidiana"] is not None else "-"
                lat_str = f"{item['latencia_ms']:.0f} ms" if item["latencia_ms"] is not None else "-"
                tabela.insert("", "end", values=(
                    item["data_hora"],
                    item["conta_destino"],
                    self._formatar_brl(item["valor"]),
                    item["nivel_seguranca"],
                    dist_str,
                    lat_str,
                    item["status"]
                ))

        for texto_btn, val_filtro in [("Todas", "TODOS"), ("Aprovadas", "APROVADA"), ("Bloqueadas", "BLOQUEADA")]:
            tk.Button(
                barra_extrato, text=texto_btn, font=("Arial", 9),
                bg="#334155", fg="white", padx=10, pady=3, bd=0, cursor="hand2",
                command=lambda v=val_filtro: [filtro_var.set(v), carregar_tabela()]
            ).pack(side="right", padx=3)

        carregar_tabela()
        tabela.pack(fill="both", expand=True)

    # =========================================================================
    # JANELA MODAL: VALIDAÇÃO FACIAL AO VIVO (NÍVEL 2 E NÍVEL 3 COFRE MÁXIMO)
    # =========================================================================
    def abrir_modal_validacao_facial(self, conta_destino: str, nome_dest: str, valor: float, nivel: int):
        modal = tk.Toplevel(self.root)
        titulo_nivel = "NÍVEL 2 - ALTO VALOR" if nivel == 2 else "NÍVEL 3 - COFRE MÁXIMO (MULTIFATOR)"
        modal.title(f"Barreira Biométrica - {titulo_nivel}")
        modal.geometry("600x665" if nivel == 3 else "600x590")
        modal.configure(bg="#1E293B")
        modal.transient(self.root)
        modal.grab_set()

        cor_banner = "#B45309" if nivel == 2 else "#991B1B"
        tk.Label(
            modal,
            text=f"{titulo_nivel} ({self._formatar_brl(valor)})",
            font=("Arial", 12, "bold"),
            bg=cor_banner,
            fg="#FFFFFF",
            pady=10
        ).pack(fill="x")

        instrucao = (
            f"Destinatário: {nome_dest} ({conta_destino})\n"
            + ("Posicione o rosto e clique em Validar." if nivel == 2
               else "Cofre Máximo: Pisque diante da câmera (Prova de Vida) e confirme sua senha de acesso.")
        )
        tk.Label(modal, text=instrucao, font=("Arial", 10), bg="#1E293B", fg="#E2E8F0", pady=8).pack()

        lbl_cam = tk.Label(modal, bg="#000000", width=480, height=330)
        lbl_cam.pack(pady=4)

        estado_liveness = {"piscada_detectada": False, "ear_atual": 0.30}
        entry_senha_n3 = None

        if nivel == 3:
            lbl_liveness = tk.Label(
                modal,
                text="Prova de Vida (EAR): Aguardando piscada ocular...",
                font=("Arial", 10, "bold"),
                bg="#0F172A",
                fg="#FBBF24",
                pady=6,
                padx=12
            )
            lbl_liveness.pack(fill="x", padx=30, pady=(6, 4))

            frame_s3 = tk.Frame(modal, bg="#1E293B")
            frame_s3.pack(fill="x", padx=30, pady=4)
            tk.Label(frame_s3, text="Reconfirme sua Senha:", font=("Arial", 10, "bold"), bg="#1E293B", fg="#E2E8F0").pack(side="left")
            entry_senha_n3 = tk.Entry(frame_s3, font=("Arial", 11), show="*", width=22, bg="#0F172A", fg="white", insertbackground="white")
            entry_senha_n3.pack(side="right", ipady=3)

        lbl_feedback = tk.Label(
            modal,
            text="Sensor 640x480 ativo. Aguardando validação biométrica...",
            font=("Arial", 10, "bold"),
            bg="#0F172A",
            fg="#38BDF8",
            pady=10,
            padx=15,
            wraplength=520
        )
        lbl_feedback.pack(fill="x", padx=30, pady=10)

        if not self.pipeline.iniciar_camera():
            messagebox.showerror("Erro de Câmera", "Webcam indisponível. Operação abortada.", parent=modal)
            modal.destroy()
            return

        estado = {"ativo": True, "frame_atual": None, "contador_frames": 0}

        def fechar_modal():
            estado["ativo"] = False
            self.pipeline.liberar_camera()
            modal.destroy()

        modal.protocol("WM_DELETE_WINDOW", fechar_modal)

        def loop_camera():
            if not estado["ativo"]:
                return
            ok, frame = self.pipeline.capturar_frame()
            if ok and frame is not None:
                estado["frame_atual"] = frame
                estado["contador_frames"] += 1
                exibicao = frame.copy()

                if nivel == 3 and not estado_liveness["piscada_detectada"] and estado["contador_frames"] % 3 == 0:
                    rgb_nat, rgb_red, bgr_eq = self.pipeline.preprocessar_frame(frame)
                    locs, rgb_usado = self.pipeline.detectar_rostos_adaptativo(rgb_nat, rgb_red, bgr_eq)
                    if len(locs) == 1:
                        lms = face_recognition.face_landmarks(rgb_usado, face_locations=locs)
                        if lms:
                            ear = self.pipeline.calcular_liveness_ear(lms[0])
                            estado_liveness["ear_atual"] = ear
                            if ear < self.pipeline.LIMIAR_EAR_PISCADA:
                                estado_liveness["piscada_detectada"] = True
                                lbl_liveness.configure(
                                    text=f"Prova de Vida (EAR = {ear:.2f}): PISCADA CONFIRMADA!",
                                    fg="#4ADE80"
                                )

                exibicao = cv2.resize(exibicao, (480, 330))
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(exibicao, cv2.COLOR_BGR2RGB)))
                lbl_cam.imgtk = img_tk
                lbl_cam.configure(image=img_tk)
            modal.after(25, loop_camera)

        def executar_matching():
            if not estado["ativo"] or estado["frame_atual"] is None:
                return

            if nivel == 3:
                if not estado_liveness["piscada_detectada"]:
                    messagebox.showwarning(
                        "Prova de Vida Pendente",
                        "Para acessar o Cofre (Nível 3), pisque os olhos diante da câmera para comprovar vivacidade.",
                        parent=modal
                    )
                    return
                if not self.db.verificar_senha_adicional(self.usuario_logado["numero_conta"], entry_senha_n3.get()):
                    messagebox.showerror("Senha Inválida", "A reconfirmação de senha falhou.", parent=modal)
                    return

            lbl_feedback.configure(text="Executando pipeline ResNet-34 (68 marcos + embedding 128D)...", fg="#FBBF24")
            modal.update_idletasks()

            resultado = self.pipeline.verificar_identidade(
                frame_bgr=estado["frame_atual"].copy(),
                embedding_cadastrado=self.usuario_logado["embedding"]
            )

            cor_box = (74, 222, 128) if resultado["aprovado"] else (0, 0, 255)
            frame_anotado = self.pipeline.desenhar_hud_biometrico(
                estado["frame_atual"], resultado["roi"], resultado["landmarks"], cor_bgr=cor_box
            )
            exibicao = cv2.resize(frame_anotado, (480, 330))
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(exibicao, cv2.COLOR_BGR2RGB)))
            lbl_cam.imgtk = img_tk
            lbl_cam.configure(image=img_tk)

            if not resultado["rosto_detectado"]:
                lbl_feedback.configure(text=resultado["mensagem"], fg="#FBBF24")
                return

            distancia = resultado["distancia"]
            sim_cos = resultado["similaridade_cosseno"]
            latencia_ms = resultado["latencia_ms"]
            nome_nivel = "Nível 2 (Facial)" if nivel == 2 else "Nível 3 (Cofre)"

            if resultado["aprovado"]:
                lbl_feedback.configure(
                    text=f"{resultado['mensagem']} | Latência: {latencia_ms:.0f} ms",
                    fg="#4ADE80"
                )
                modal.update()
                modal.after(700)

                sucesso, msg = self.db.executar_transferencia(
                    conta_origem=self.usuario_logado["numero_conta"],
                    conta_destino=conta_destino,
                    valor=valor,
                    nivel_seguranca=nome_nivel,
                    exigiu_biometria=True,
                    liveness_aprovado=(nivel == 3),
                    distancia_euclidiana=distancia,
                    similaridade_cosseno=sim_cos,
                    latencia_ms=latencia_ms
                )
                fechar_modal()
                if sucesso:
                    messagebox.showinfo(
                        "Acesso ao Cofre / Transação Autorizada",
                        f"{msg}\n\n• Nível: {nome_nivel}\n• Distância Euclidiana: {distancia:.4f} (<= 0.55)\n"
                        f"• Confiança Estimada: {resultado['confianca_pct']:.1f}%\n• Latência Total: {latencia_ms:.0f} ms"
                    )
                    self.exibir_painel_bancario()
                else:
                    messagebox.showerror("Erro", msg)
            else:
                lbl_feedback.configure(
                    text=f"{resultado['mensagem']} | Latência: {latencia_ms:.0f} ms",
                    fg="#F87171"
                )
                self.db.registrar_tentativa_bloqueada(
                    conta_origem=self.usuario_logado["numero_conta"],
                    conta_destino=conta_destino,
                    valor=valor,
                    nivel_seguranca=nome_nivel,
                    distancia_euclidiana=distancia,
                    similaridade_cosseno=sim_cos,
                    latencia_ms=latencia_ms,
                    motivo=f"Distância {distancia:.4f} > 0.55"
                )
                modal.update()
                modal.after(1100)
                fechar_modal()
                messagebox.showerror(
                    "Acesso Negado por Segurança",
                    f"Biometria facial rejeitada.\n• Distância obtida: {distancia:.4f}\n• Limiar máximo: 0.5500"
                )
                self.exibir_painel_bancario()

        tk.Button(
            modal, text="VALIDAR IDENTIDADE E CONFIRMAR", font=("Arial", 10, "bold"),
            bg="#16A34A", fg="white", pady=9, width=42, bd=0, cursor="hand2",
            command=executar_matching
        ).pack(pady=(2, 8))

        loop_camera()

    # =========================================================================
    # EXTRA APS: INSPETOR VISUAL DAS 5 FASES DO PROCESSAMENTO DE IMAGENS DIGITAIS
    # =========================================================================
    def abrir_inspetor_pdi_5_fases(self):
        win = tk.Toplevel(self.root)
        win.title("Inspetor Técnico - As 5 Fases do Processamento de Imagens Digitais (PIVC)")
        win.geometry("960x560")
        win.configure(bg="#0F172A")
        win.transient(self.root)
        win.grab_set()

        tk.Label(
            win,
            text="PIPELINE DE VISÃO COMPUTACIONAL - DEMONSTRAÇÃO DAS 5 FASES DO PDI",
            font=("Arial", 13, "bold"),
            bg="#1E293B",
            fg="#38BDF8",
            pady=12
        ).pack(fill="x")

        grid = tk.Frame(win, bg="#0F172A", padx=20, pady=15)
        grid.pack(fill="both", expand=True)

        paineis = {}
        titulos = [
            ("f1", "1. Aquisição (Sensor 640x480 BGR)"),
            ("f2", "2. Pré-processamento (CIELAB + CLAHE)"),
            ("f3", "3. Segmentação (ROI Detector HOG)"),
            ("f4", "4. Extração (68 Marcos + ResNet-34 128D)"),
        ]

        for idx, (chave, titulo) in enumerate(titulos):
            r, c = divmod(idx, 2)
            box = tk.Frame(grid, bg="#1E293B", padx=10, pady=8)
            box.grid(row=r, column=c, padx=10, pady=8)
            tk.Label(box, text=titulo, font=("Arial", 10, "bold"), bg="#1E293B", fg="#E2E8F0").pack(pady=(0, 5))
            lbl = tk.Label(box, bg="#000000", width=260, height=175)
            lbl.pack()
            paineis[chave] = lbl

        lbl_fase5 = tk.Label(
            win,
            text="5. Classificação & Telemetria: Clique em 'Capturar e Analisar Quadro' para inspecionar.",
            font=("Arial", 10, "bold"),
            bg="#1E293B",
            fg="#4ADE80",
            pady=10
        )
        lbl_fase5.pack(fill="x", padx=30, pady=(0, 10))

        if not self.pipeline.iniciar_camera():
            messagebox.showerror("Erro", "Webcam indisponível.", parent=win)
            win.destroy()
            return

        def fechar_inspetor():
            self.pipeline.liberar_camera()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", fechar_inspetor)

        def analisar_5_fases():
            ok, frame = self.pipeline.capturar_frame()
            if not ok or frame is None:
                return

            extracao = self.pipeline.extrair_caracteristicas_completas(frame)
            resultado = self.pipeline.verificar_identidade(frame, self.usuario_logado["embedding"])

            img1 = cv2.resize(frame, (260, 175))
            img2 = cv2.resize(extracao["frame_equalizado"], (260, 175))

            img3_raw = frame.copy()
            if extracao["roi"] is not None:
                t, r, b, l = extracao["roi"]
                cv2.rectangle(img3_raw, (l, t), (r, b), (0, 255, 255), 3)
            img3 = cv2.resize(img3_raw, (260, 175))

            img4_raw = self.pipeline.desenhar_hud_biometrico(frame, extracao["roi"], extracao["landmarks"])
            img4 = cv2.resize(img4_raw, (260, 175))

            for chave, img_bgr in [("f1", img1), ("f2", img2), ("f3", img3), ("f4", img4)]:
                tk_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
                paineis[chave].imgtk = tk_img
                paineis[chave].configure(image=tk_img)

            if resultado["rosto_detectado"]:
                tm = resultado["tempos_ms"]
                lbl_fase5.configure(
                    text=(
                        f"5. Classificação: {resultado['mensagem']}  |  "
                        f"Tempos: Prep={tm['preprocessamento']:.1f}ms, Seg={tm['segmentacao']:.1f}ms, "
                        f"Ext={tm['extracao']:.1f}ms, Total={resultado['latencia_ms']:.1f}ms"
                    ),
                    fg="#4ADE80" if resultado["aprovado"] else "#F87171"
                )
            else:
                lbl_fase5.configure(text=f"5. Classificação: {resultado['mensagem']}", fg="#FBBF24")

        tk.Button(
            win, text="CAPTURAR E ANALISAR AS 5 FASES DO PDI", font=("Arial", 10, "bold"),
            bg="#0284C7", fg="white", pady=8, padx=20, bd=0, cursor="hand2",
            command=analisar_5_fases
        ).pack(pady=(0, 12))


if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaBancarioApp(root)
    root.mainloop()