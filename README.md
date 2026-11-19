# APS 6° Semestre

Projeto desenvolvido para a disciplina de **Processamento de Imagens e Visão Computacional** (6º Semestre de Ciências da Computação — UNIP | Profa. Stephany Oliveira)[cite: 1, 4].

## Visão Geral
Aplicação desktop em Python que implementa autenticação biométrica facial adaptativa orientada a risco para transferências financeiras[cite: 1]:
- **Nível 1 (< R$ 1.000,00):** Liberação imediata após autenticação primária (`bcrypt`)[cite: 1, 2, 4].
- **Nível 2 (R$ 1.000,00 a R$ 9.999,99):** Bloqueio automático e validação facial 1:1 em tempo real (Distância Euclidiana $\le 0,55$)[cite: 1, 2, 4].
- **Nível 3 ($\ge$ R$ 10.000,00):** Autenticação multifator com reconhecimento facial 1:1, prova de vida ativa por piscada ocular (EAR) e reconfirmação de senha[cite: 3, 4].

## Stack Tecnológica
- **Linguagem:** Python 3.10+[cite: 1]
- **Visão Computacional:** OpenCV (`cv2`), `face_recognition` / `dlib` (HOG + ResNet-34 128D)[cite: 1, 2, 3]
- **Segurança e Persistência:** SQLite3, `bcrypt` e SHA-256 (armazenamento exclusivo de vetores 128D em conformidade com a LGPD)[cite: 1, 2]
- **Interface Gráfica:** Tkinter + Pillow[cite: 1]

## Como Executar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Inicie o sistema:
   ```bash
   python main_app.py
   ```

## Contas de demonstração

As contas abaixo são criadas automaticamente pelo banco de dados. A senha inicial de todas é `234`:

| Usuário | Conta | Chave Pix |
|---|---|---|
| Lucas | `1003-0` | Lucas |
| Igor | `1002-0` | Igor |
| Guilherme | `1005-0` | Guilherme |
| Gustavo | `1001-0` | Gustavo |
| Moisés | `1004-0` | Moisés |

Após entrar, use **Cadastrar Minha Biometria** no painel para capturar e vincular as três amostras faciais. Também é possível informar o nome acima como chave Pix ou o número da conta.