import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import cv2
import numpy as np
import sqlite3
import json
import bcrypt
import random
import os
import io
from datetime import datetime
from PIL import Image

# ==========================================
# UNIPay v12 - Biometria Nativa via OpenCV
# ==========================================

UNIP_BLUE = "#002B5C"
UNIP_YELLOW = "#FFC72C"
BG_COLOR = "#F4F6F9"
DB_NAME = "unipay_v12.db" 

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Carrega o detetor de rostos nativo do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def formatar_moeda(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def gerar_numero_cartao():
    return f"5432 {random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"

def gerar_conta():
    return f"{random.randint(10000, 99999)}-{random.randint(0, 9)}"

def inicializar_banco():
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pin_hash BLOB NOT NULL,
            balance REAL NOT NULL,
            face_encoding TEXT,
            conta TEXT NOT NULL,
            cartao TEXT NOT NULL,
            profile_pic TEXT,
            profile_image BLOB
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS transacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            tipo TEXT,
            valor REAL,
            data TEXT,
            descricao TEXT,
            FOREIGN KEY(user_id) REFERENCES usuarios(id)
        )''')
        
        cur.execute("SELECT COUNT(*) FROM usuarios")
        if cur.fetchone()[0] == 0:
            pin_hash = bcrypt.hashpw(b"1234", bcrypt.gensalt())
            for nome in ["lucas", "gustavo", "guilherme", "moises", "higor"]:
                cur.execute("INSERT INTO usuarios (username,pin_hash,balance,face_encoding,conta,cartao,profile_pic,profile_image) VALUES (?,?,?,?,?,?,?,?)",
                            (nome, pin_hash, 10000.0, None, gerar_conta(), gerar_numero_cartao(), "", None))
                cur.execute("INSERT INTO transacoes (user_id,tipo,valor,data,descricao) VALUES (?,?,?,?,?)",
                            (cur.lastrowid, "ENTRADA", 10000.0, datetime.now().strftime("%d/%m/%Y %H:%M"), "Depósito Inicial / Boas-vindas"))

class PinDialog(ctk.CTkToplevel):
    def __init__(self, master, pin_hash, titulo="Validação"):
        super().__init__(master)
        self.title("Segurança UNIPay")
        self.geometry("350x250")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.pin_hash = pin_hash
        self.autenticado = False
        
        ctk.CTkLabel(self, text="🔒", font=("Segoe UI", 30)).pack(pady=(18, 0))
        ctk.CTkLabel(self, text=titulo, font=("Segoe UI", 14, "bold"), text_color=UNIP_BLUE).pack(pady=8)
        self.entry = ctk.CTkEntry(self, show="*", width=160, justify="center")
        self.entry.pack(pady=8)
        self.entry.focus()
        ctk.CTkButton(self, text="Confirmar", command=self.verificar).pack(pady=12)

    def verificar(self):
        try:
            self.autenticado = bcrypt.checkpw(self.entry.get().strip().encode(), self.pin_hash)
        except (ValueError, TypeError):
            self.autenticado = False
            
        if not self.autenticado:
            messagebox.showerror("Erro", "Senha incorreta!", parent=self)
        self.destroy()

class UNIPayApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("UNIPay - O Banco da UNIP")
        self.geometry("900x750")
        self.configure(fg_color=BG_COLOR)
        
        self.usuario_atual = None
        self.dados_cartao_visiveis = False
        self.foto_header = self.foto_perfil = None
        self.valor_mensalidade = 1250.00
        self.valor_fatura = 350.50
        
        inicializar_banco()
        self.tela_login()

    def limpar_tela(self):
        for w in self.winfo_children():
            w.destroy()

    def buscar_usuario(self, user_id):
        with sqlite3.connect(DB_NAME) as conn:
            return conn.execute("SELECT id,username,pin_hash,balance,face_encoding,conta,cartao,profile_pic,profile_image FROM usuarios WHERE id=?", (user_id,)).fetchone()

    def carregar_usuario(self, row):
        self.usuario_atual = {
            "id": row[0], "username": row[1], "pin_hash": row[2], "balance": row[3],
            "face_encoding": json.loads(row[4]) if row[4] else None,
            "conta": row[5], "cartao": row[6], "profile_pic": row[7], "profile_image": row[8]
        }
        self.foto_header = self.foto_perfil = None
        data = self.usuario_atual["profile_image"]
        
        try:
            pil = None
            if data:
                pil = Image.open(io.BytesIO(data)).convert("RGB")
            elif row[7] and os.path.exists(row[7]):
                pil = Image.open(row[7]).convert("RGB")
                
            if pil:
                self.foto_header = ctk.CTkImage(light_image=pil, size=(45,45))
                self.foto_perfil = ctk.CTkImage(light_image=pil, size=(120,120))
        except Exception as e:
            print("Falha ao abrir foto de perfil:", e)

    def tela_login(self):
        self.limpar_tela()
        top = ctk.CTkFrame(self, fg_color=UNIP_BLUE, corner_radius=0, height=220)
        top.pack(fill="x")
        ctk.CTkLabel(top, text="UNIPay", font=("Segoe UI Black",48), text_color=UNIP_YELLOW).pack(pady=(48,5))
        ctk.CTkLabel(top, text="O Banco da Universidade Paulista", text_color="white").pack()
        
        card = ctk.CTkFrame(self, fg_color="white", corner_radius=15, width=450, height=300)
        card.pack(pady=50, expand=True); card.pack_propagate(False)
        ctk.CTkLabel(card, text="Acesso à Conta", font=("Segoe UI",22,"bold"), text_color=UNIP_BLUE).pack(pady=(30,20))
        ctk.CTkButton(card, text="ENTRAR COM BIOMETRIA", fg_color=UNIP_YELLOW, text_color=UNIP_BLUE, height=48, command=self.iniciar_login_biometrico).pack(fill="x", padx=45, pady=10)
        ctk.CTkLabel(card, text="OU", text_color="gray").pack(pady=3)
        ctk.CTkButton(card, text="ENTRAR COM SENHA", fg_color=UNIP_BLUE, height=48, command=self.tela_login_senha).pack(fill="x", padx=45, pady=10)

    def tela_login_senha(self):
        self.limpar_tela()
        ctk.CTkLabel(self, text="UNIPay | Acesso Seguro", font=("Segoe UI Black",24), text_color=UNIP_BLUE).pack(pady=35)
        
        card = ctk.CTkFrame(self, fg_color="white", width=400)
        card.pack(pady=20, expand=True)
        ctk.CTkLabel(card, text="Usuário", text_color=UNIP_BLUE).pack(pady=(25,5))
        self.entry_user = ctk.CTkEntry(card, width=300, height=42); self.entry_user.pack(pady=5)
        ctk.CTkLabel(card, text="Senha (PIN)", text_color=UNIP_BLUE).pack(pady=(15,5))
        self.entry_pin = ctk.CTkEntry(card, width=300, height=42, show="*"); self.entry_pin.pack(pady=5)
        
        ctk.CTkButton(card, text="Acessar Conta", width=300, command=self.verificar_login).pack(pady=18)
        ctk.CTkButton(card, text="Voltar", fg_color="transparent", text_color="gray", command=self.tela_login).pack(pady=5)

    def verificar_login(self):
        with sqlite3.connect(DB_NAME) as conn:
            row = conn.execute("SELECT id,username,pin_hash,balance,face_encoding,conta,cartao,profile_pic,profile_image FROM usuarios WHERE username=?", (self.entry_user.get().strip().lower(),)).fetchone()
        
        try:
            ok = row and bcrypt.checkpw(self.entry_pin.get().strip().encode(), row[2])
        except (ValueError, TypeError):
            ok = False
            
        if not ok:
            return messagebox.showerror("Erro", "Usuário ou senha incorretos.")
            
        self.carregar_usuario(row)
        self.tela_dashboard()
        
        if not self.usuario_atual["face_encoding"]:
            if messagebox.askyesno("Ativar biometria", "Login realizado. Deseja cadastrar agora sua foto/biometria facial?"):
                self.tabs.set("Meu Perfil")

    def obter_encoding(self, image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        if len(faces) != 1:
            raise ValueError("A imagem precisa conter exatamente UM rosto visível e bem iluminado.")
            
        x, y, w, h = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        
        face_resized = cv2.resize(face_crop, (100, 100))
        return face_resized.tolist()

    def iniciar_login_biometrico(self):
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.execute("SELECT id,username,pin_hash,balance,face_encoding,conta,cartao,profile_pic,profile_image FROM usuarios WHERE face_encoding IS NOT NULL").fetchall()
        
        if not rows:
            return messagebox.showinfo("Aviso", "Nenhuma biometria cadastrada no sistema. Entre com a senha primeiro.")
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return messagebox.showerror("Erro", "Não foi possível abrir a câmera.")
            
        matched = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                
                for (x, y, w, h) in faces:
                    face_crop = gray[y:y+h, x:x+w]
                    test_face = cv2.resize(face_crop, (100, 100))
                    
                    for row in rows:
                        known_face = np.array(json.loads(row[4]), dtype=np.uint8)
                        similarity = cv2.matchTemplate(test_face, known_face, cv2.TM_CCOEFF_NORMED)[0][0]
                        
                        if similarity > 0.45: 
                            matched = row
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"IDENTIFICADO: {row[1].upper()}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            break
                    if matched: break
                        
                cv2.putText(frame,"Olhe para a camera | Q para cancelar",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2)
                cv2.imshow("Login biometrico UNIPay",frame)
                
                if matched:
                    cv2.waitKey(700)
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            messagebox.showerror("Erro de biometria", str(e))
        finally:
            cap.release(); cv2.destroyAllWindows()
            
        if matched:
            self.carregar_usuario(matched)
            self.tela_dashboard()

    def tela_dashboard(self):
        self.limpar_tela()
        header = ctk.CTkFrame(self, fg_color=UNIP_BLUE, corner_radius=0, height=90); header.pack(fill="x")
        
        if self.foto_header:
            ctk.CTkLabel(header, text="", image=self.foto_header).pack(side="left", padx=(25,8), pady=18)
        else:
            ctk.CTkLabel(header, text="👤", font=("Segoe UI",28), text_color="white").pack(side="left", padx=(25,8), pady=18)
            
        ctk.CTkLabel(header, text=f"UNIPay | Olá, {self.usuario_atual['username'].capitalize()}!", font=("Segoe UI",19,"bold"), text_color=UNIP_YELLOW).pack(side="left")
        ctk.CTkButton(header, text="Sair", width=75, fg_color="#e53935", command=self.tela_login).pack(side="right", padx=25)
        
        self.tabs = ctk.CTkTabview(self, fg_color="white", segmented_button_selected_color=UNIP_BLUE)
        self.tabs.pack(padx=18,pady=18,fill="both",expand=True)
        for name in ["Resumo","Pix","Mensalidade","Cartão","Extrato","Meu Perfil"]: self.tabs.add(name)
        
        self.render_resumo(); self.render_pix(); self.render_mensalidade(); self.render_cartao(); self.render_extrato(); self.render_perfil()

    def render_resumo(self):
        f = self.tabs.tab("Resumo")
        ctk.CTkLabel(f,text="Saldo Disponível",text_color="gray").pack(anchor="w",padx=30,pady=(35,0))
        self.lbl_saldo=ctk.CTkLabel(f,text=formatar_moeda(self.usuario_atual["balance"]),font=("Segoe UI Black",38),text_color="#2e7d32")
        self.lbl_saldo.pack(anchor="w",padx=30)

    def render_pix(self):
        f = self.tabs.tab("Pix")
        ctk.CTkLabel(f,text="Área Pix",font=("Segoe UI",24,"bold"),text_color=UNIP_BLUE).pack(pady=18)
        ctk.CTkLabel(f,text="Conta de destino (usuário)").pack()
        self.entry_destino=ctk.CTkEntry(f,width=330); self.entry_destino.pack(pady=5)
        ctk.CTkLabel(f,text="Valor (R$)").pack(pady=(10,0))
        self.entry_valor_pix=ctk.CTkEntry(f,width=330); self.entry_valor_pix.pack(pady=5)
        ctk.CTkButton(f,text="Transferir Agora",fg_color=UNIP_BLUE,width=330,height=45,command=self.processar_pix).pack(pady=20)

    def render_mensalidade(self):
        f = self.tabs.tab("Mensalidade")
        for w in f.winfo_children(): w.destroy() # Limpa widgets anteriores caso haja atualização
        
        ctk.CTkLabel(f,text="Gestão Acadêmica UNIP",font=("Segoe UI",23,"bold"),text_color=UNIP_BLUE).pack(pady=35)
        if self.valor_mensalidade > 0:
            ctk.CTkLabel(f,text=f"Total a pagar: {formatar_moeda(self.valor_mensalidade)}",font=("Segoe UI",18,"bold"),text_color="#d32f2f").pack(pady=15)
            ctk.CTkButton(f,text="Pagar",fg_color=UNIP_YELLOW,text_color=UNIP_BLUE,command=self.pagar_mensalidade).pack(pady=10)
        else:
            ctk.CTkLabel(f,text="Mensalidade paga deste mês",font=("Segoe UI", 18, "bold"), text_color="#2e7d32").pack(pady=30)

    def render_cartao(self):
        f = self.tabs.tab("Cartão")
        for w in f.winfo_children(): w.destroy() # Limpa widgets anteriores caso haja atualização
        
        num = self.usuario_atual["cartao"] if self.dados_cartao_visiveis else "**** **** **** "+self.usuario_atual["cartao"][-4:]
        ctk.CTkLabel(f,text="UNIPay Credit",font=("Segoe UI Black",22),text_color=UNIP_BLUE).pack(pady=(35,15))
        ctk.CTkLabel(f,text=num,font=("Courier",22,"bold")).pack(pady=20)
        ctk.CTkLabel(f,text=self.usuario_atual["username"].upper()).pack(pady=5)
        ctk.CTkButton(f,text="Ocultar dados" if self.dados_cartao_visiveis else "Mostrar dados",command=self.toggle_cartao).pack(pady=15)
        
        if self.valor_fatura > 0:
            ctk.CTkButton(f,text=f"Pagar fatura de {formatar_moeda(self.valor_fatura)}",command=self.pagar_fatura).pack(pady=8)
        else:
            ctk.CTkLabel(f,text="Fatura paga deste mês",font=("Segoe UI", 16, "bold"), text_color="#2e7d32").pack(pady=8)

    def toggle_cartao(self):
        self.dados_cartao_visiveis = not self.dados_cartao_visiveis
        self.render_cartao()

    def render_extrato(self):
        f = self.tabs.tab("Extrato")
        for w in f.winfo_children(): w.destroy()
        box = ctk.CTkScrollableFrame(f,fg_color="transparent"); box.pack(fill="both",expand=True,padx=15,pady=12)
        with sqlite3.connect(DB_NAME) as conn:
            tx = conn.execute("SELECT tipo,valor,data,descricao FROM transacoes WHERE user_id=? ORDER BY id DESC",(self.usuario_atual["id"],)).fetchall()
        for tipo,valor,data,desc in tx:
            entrada = (tipo == "ENTRADA")
            row = ctk.CTkFrame(box,fg_color="#e8f5e9" if entrada else "#f7f7f7"); row.pack(fill="x",pady=4)
            ctk.CTkLabel(row,text=f"{desc}\n{data}",justify="left").pack(side="left",padx=12,pady=8)
            ctk.CTkLabel(row,text=f"{'+' if entrada else '-'} {formatar_moeda(valor)}",text_color="#2e7d32" if entrada else "#222",font=("Segoe UI",14,"bold")).pack(side="right",padx=12)

    def render_perfil(self):
        f = self.tabs.tab("Meu Perfil")
        for w in f.winfo_children(): w.destroy()
        
        info = ctk.CTkFrame(f,fg_color=BG_COLOR,corner_radius=12); info.pack(fill="x",padx=20,pady=20)
        
        if self.foto_perfil:
            ctk.CTkLabel(info,text="",image=self.foto_perfil).pack(pady=(18,8))
        else:
            ctk.CTkLabel(info,text="👤",font=("Segoe UI",70),text_color="gray").pack(pady=(18,8))
            
        ctk.CTkLabel(info,text=f"Conta: {self.usuario_atual['conta']} | Agência: 0001").pack()
        ctk.CTkLabel(info,text="Biometria cadastrada e ativa" if self.usuario_atual["face_encoding"] else "Conta sem foto. Registre abaixo.",text_color="#2e7d32" if self.usuario_atual["face_encoding"] else "#b26a00").pack(pady=6)
        
        buttons = ctk.CTkFrame(info,fg_color="transparent"); buttons.pack(pady=16)
        ctk.CTkButton(buttons,text="Tirar foto (Webcam)",fg_color=UNIP_BLUE,command=lambda:self.cadastrar_foto("camera")).pack(side="left",padx=6)
        ctk.CTkButton(buttons,text="Escolher do Computador",fg_color=UNIP_YELLOW,text_color=UNIP_BLUE,command=lambda:self.cadastrar_foto("arquivo")).pack(side="left",padx=6)
        
        ctk.CTkLabel(f,text="Sua foto é salva utilizando o motor biométrico nativo do OpenCV.",font=("Segoe UI",11),text_color="gray").pack()

    def cadastrar_foto(self, metodo):
        image_rgb = None
        
        if metodo == "arquivo":
            path = filedialog.askopenfilename(title="Escolha uma foto",filetypes=[("Imagens","*.jpg *.jpeg *.png")])
            if not path: return
            try: 
                img = cv2.imread(path)
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e: 
                return messagebox.showerror("Erro",f"Não foi possível abrir a imagem: {e}")
                
        elif metodo == "camera":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened(): 
                return messagebox.showerror("Erro","Não foi possível abrir a câmera.")
                
            messagebox.showinfo("Câmera","Centralize seu rosto. Pressione ESPAÇO a qualquer momento para capturar; Q cancela.")
            captured = None
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    clean_frame = frame.copy() 
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                    
                    for (x, y, w, h) in faces: 
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                    cv2.putText(frame,"ESPACO captura | Q cancela",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                    cv2.imshow("Foto de perfil UNIPay",frame)
                    
                    key = cv2.waitKey(1)&0xFF
                    if key == ord(' '):
                        if len(faces) == 1:
                            captured = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB) 
                            cv2.putText(frame,"FOTO CAPTURADA!",(50,100),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)
                            cv2.imshow("Foto de perfil UNIPay",frame)
                            cv2.waitKey(600)
                            break
                        else:
                            messagebox.showwarning("Atenção", "Fique sozinho e de frente para a câmera.")
                    if key == ord('q'): 
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
                
            image_rgb = captured

        if image_rgb is None: return

        try:
            encoding = self.obter_encoding(image_rgb)
            pil = Image.fromarray(image_rgb).convert("RGB")
            pil.thumbnail((600,600)) 
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=88)
            image_blob = buf.getvalue()
            
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("UPDATE usuarios SET face_encoding=?, profile_image=?, profile_pic=NULL WHERE id=?",
                             (json.dumps(encoding), sqlite3.Binary(image_blob), self.usuario_atual["id"]))
                conn.commit()
                
            self.carregar_usuario(self.buscar_usuario(self.usuario_atual["id"]))
            messagebox.showinfo("Sucesso","Foto salva! Biometria ativada no banco de dados.")
            self.tela_dashboard()
            self.tabs.set("Meu Perfil")
        except Exception as e:
            messagebox.showerror("Cadastro não concluído", str(e))

    def validar_rosto_transacao(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erro","Não foi possível abrir a câmera."); return False
            
        known_face = np.array(self.usuario_atual["face_encoding"], dtype=np.uint8)
        ok = False
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                
                for (x, y, w, h) in faces:
                    face_crop = gray[y:y+h, x:x+w]
                    test_face = cv2.resize(face_crop, (100, 100))
                    
                    similarity = cv2.matchTemplate(test_face, known_face, cv2.TM_CCOEFF_NORMED)[0][0]
                    
                    if similarity > 0.45: 
                        ok=True; break
                        
                cv2.putText(frame,"Validacao facial | Q cancela",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,220,255),2)
                cv2.imshow("Validacao UNIPay",frame)
                
                if ok: 
                    cv2.waitKey(600); break
                if cv2.waitKey(1)&0xFF==ord('q'): 
                    break
        finally:
            cap.release(); cv2.destroyAllWindows()
            
        if not ok: 
            messagebox.showerror("Biometria","Rosto não reconhecido. Transação cancelada.")
        return ok

    def processar_pix(self):
        destino = self.entry_destino.get().strip().lower()
        if not destino: return messagebox.showerror("Erro","Informe o usuário de destino.")
        
        try: valor = float(self.entry_valor_pix.get().strip().replace(",","."))
        except ValueError: return messagebox.showerror("Erro","Valor inválido.")
        
        if destino == self.usuario_atual["username"]: return messagebox.showerror("Erro","Não é possível transferir para sua própria conta.")
        
        with sqlite3.connect(DB_NAME) as conn:
            row = conn.execute("SELECT id FROM usuarios WHERE username=?",(destino,)).fetchone()
            
        if not row: return messagebox.showerror("Erro","Conta não encontrada.")
        
        # Passa o nome de quem vai receber na descrição de saída para o histórico
        if self.processar_transacao(valor, f"Pix enviado para {destino.capitalize()}", destino):
            self.entry_destino.delete(0,tk.END); self.entry_valor_pix.delete(0,tk.END)

    def processar_transacao(self,valor,descricao,dest_username=None):
        if valor<=0 or valor>self.usuario_atual["balance"]:
            messagebox.showerror("Erro","Valor inválido ou saldo insuficiente."); return False
            
        if valor>1000:
            if self.usuario_atual["face_encoding"]:
                if not self.validar_rosto_transacao(): return False
            else:
                messagebox.showinfo("Biometria","Biometria não cadastrada; será solicitada a senha.")
                d = PinDialog(self,self.usuario_atual["pin_hash"]); self.wait_window(d)
                if not d.autenticado: return False
        else:
            d = PinDialog(self,self.usuario_atual["pin_hash"]); self.wait_window(d)
            if not d.autenticado: return False
            
        data = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("UPDATE usuarios SET balance=balance-? WHERE id=? AND balance>=?",(valor,self.usuario_atual["id"],valor))
            if cur.rowcount!=1: return False
            
            cur.execute("INSERT INTO transacoes (user_id,tipo,valor,data,descricao) VALUES (?,?,?,?,?)",(self.usuario_atual["id"],"SAIDA",valor,data,descricao))
            
            if dest_username:
                dest = cur.execute("SELECT id FROM usuarios WHERE username=?",(dest_username,)).fetchone()
                if not dest: raise ValueError("Conta de destino não encontrada.")
                cur.execute("UPDATE usuarios SET balance=balance+? WHERE id=?",(valor,dest[0]))
                
                # Registra o histórico formatado de quem está recebendo o dinheiro
                cur.execute("INSERT INTO transacoes (user_id,tipo,valor,data,descricao) VALUES (?,?,?,?,?)",
                            (dest[0], "ENTRADA", valor, data, f"Pix recebido de {self.usuario_atual['username'].capitalize()}"))
                
        self.usuario_atual["balance"] -= valor
        self.lbl_saldo.configure(text=formatar_moeda(self.usuario_atual["balance"]))
        self.render_extrato() # Atualiza visualmente o extrato
        messagebox.showinfo("Transação concluída",f"{descricao}\nValor: {formatar_moeda(valor)}")
        return True

    def pagar_mensalidade(self):
        if self.processar_transacao(self.valor_mensalidade, "Mensalidade paga deste mês"):
            self.valor_mensalidade = 0 
            self.render_mensalidade()

    def pagar_fatura(self):
        if self.valor_fatura and self.processar_transacao(self.valor_fatura, "Fatura paga deste mês"):
            self.valor_fatura = 0 
            self.render_cartao()

if __name__ == "__main__":
    app = UNIPayApp()
    app.mainloop()