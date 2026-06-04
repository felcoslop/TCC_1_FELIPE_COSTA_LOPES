# Visualizador de Estados Operacionais - imagem estatica (nginx + gzip)
FROM nginx:alpine

# Configuracao do servidor (gzip on-the-fly + gzip_static para os CSV + cache)
COPY default.conf /etc/nginx/conf.d/default.conf

# App estatico
COPY index.html app.js style.css /usr/share/nginx/html/
COPY lib/ /usr/share/nginx/html/lib/

# Dados: apenas os CSV JA COMPRIMIDOS (.csv.gz) + o manifesto.
# (rode antes:  python comprimir_dados.py)
COPY dados/manifest.json /usr/share/nginx/html/dados/
COPY dados/*.csv.gz /usr/share/nginx/html/dados/

EXPOSE 80
