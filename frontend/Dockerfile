FROM nginx:1.27-alpine

RUN rm /etc/nginx/conf.d/default.conf

# Shared proxy headers (referenced by nginx.conf via include)
RUN printf 'proxy_set_header Host $host;\n\
proxy_set_header X-Real-IP $remote_addr;\n\
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n\
proxy_set_header X-Forwarded-Proto $scheme;\n' > /etc/nginx/proxy_params

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY src/index.html /usr/share/nginx/html/index.html

EXPOSE 8080
