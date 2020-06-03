FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    cp prestart_prod.sh prestart.sh
