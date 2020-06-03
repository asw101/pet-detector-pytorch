# DEVELOPMENT

```bash
# local dev
docker run --rm -v ${PWD}:/app/ -p 80:80 -it tiangolo/uwsgi-nginx-flask:python3.7

# local build
docker build --no-cache -t test:latest .

# local test
docker run --rm -p 80:80 -it test:latest
```
