# Deploy do Visualizador na VPS (EasyPanel)

## Ponto importante sobre os dados
O visualizador **NÃO precisa** dos GB de dados brutos do pipeline (`code/data/...`).
Ele usa só os CSV já exportados em `visualizador/dados/`, que somam **~196 MB** —
e **comprimidos com gzip viram ~39 MB**. Ou seja, "enviar para a nuvem" é mandar ~39 MB.

O nginx serve os `.csv.gz` automaticamente quando o app pede o `.csv`
(`gzip_static`), e o navegador descomprime sozinho. Nada muda no código do app.

## Arquivos de deploy (já criados nesta pasta)
- `Dockerfile` — imagem nginx servindo o site estático + CSVs comprimidos.
- `default.conf` — config do nginx (gzip, cache).
- `.dockerignore` — evita mandar os CSV crus/arquivos de dev pro build.
- `comprimir_dados.py` — gera os `dados/*.csv.gz`.

---

## Passo 1 — Gerar os dados comprimidos (uma vez, ou quando retreinar)
```
cd visualizador
python comprimir_dados.py
```
Isso cria `dados/c_*.csv.gz` (~39 MB no total). São esses que vão para a nuvem.

---

## Passo 2 — Escolha COMO subir (3 opções)

### Opção A — Git + Dockerfile (recomendada: sem registry, sem SSH)
O EasyPanel faz build a partir de um repositório Git.

1. Crie um repositório (GitHub/GitLab) contendo **o conteúdo da pasta `visualizador/`**
   na raiz (index.html, app.js, style.css, lib/, dados/manifest.json, dados/*.csv.gz,
   Dockerfile, default.conf, .dockerignore). Os `*.csv.gz` (~39 MB) cabem no Git tranquilo
   (cada arquivo < 100 MB). O `.dockerignore` impede o envio dos `.csv` crus.
   - Dica: confira o `.gitignore` do projeto para NÃO ignorar os `.csv.gz`.
2. `git add . && git commit -m "deploy visualizador" && git push`
3. No EasyPanel: **Create → App → Source = GitHub** (autorize e escolha o repo).
4. Em **Build**, selecione **Dockerfile** (Build Method = Dockerfile).
5. Em **Deploy/Ports**, exponha a porta **80**.
6. **Deploy**. O EasyPanel clona (~39 MB), builda a imagem nginx e publica.
7. Em **Domains**, associe seu domínio (o EasyPanel cuida do HTTPS via Let's Encrypt).

> Atualizar dados depois: rode `comprimir_dados.py`, `git push`, e clique em **Deploy** de novo.

### Opção B — Imagem Docker pronta (build local + registry)
Útil se não quiser os dados no Git.
```
cd visualizador
docker build -t SEU_USUARIO/visualizador:1 .
docker push SEU_USUARIO/visualizador:1     # Docker Hub ou ghcr.io
```
No EasyPanel: **Create → App → Source = Docker Image** → informe `SEU_USUARIO/visualizador:1`,
porta **80**, deploy. O `docker push` envia só ~45 MB (camadas comprimidas).

### Opção C — Volume + upload separado (dados fora da imagem)
Se quiser trocar os dados sem rebuildar a imagem:
1. No EasyPanel, adicione um **Volume** montado em `/usr/share/nginx/html/dados`.
2. Builde/deploy a imagem SEM os dados (ajuste o Dockerfile para não copiar `dados/`).
3. Suba os arquivos para o volume via SSH na VPS:
   ```
   # do seu PC (Windows: use o rsync do Git Bash/WSL, ou scp)
   rsync -avz --progress dados/*.csv.gz dados/manifest.json \
       root@SEU_IP:/etc/easypanel/projects/SEU_PROJETO/SEU_APP/volumes/dados/
   ```
   (confirme o caminho real do volume no painel do EasyPanel)
- `rsync` é resumível e só manda o que mudou — ideal se a conexão cair.

---

## Recomendação
Comece pela **Opção A**. É a mais simples: 39 MB no Git, EasyPanel builda e sobe sozinho,
com HTTPS automático. As Opções B/C servem se você quiser separar dados da imagem.

## Teste local da imagem (opcional, antes de subir)
```
cd visualizador
python comprimir_dados.py
docker build -t visualizador .
docker run --rm -p 8080:80 visualizador
# abra http://localhost:8080/
```
