FROM python:3.12-slim
WORKDIR /ragbuilder
COPY pyproject.toml requirements.lock README.md LICENSE ./
COPY src ./src
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN pip install --no-cache-dir -r requirements.lock . && useradd --create-home --uid 10001 ragbuilder && chown -R ragbuilder:ragbuilder /ragbuilder
USER ragbuilder
EXPOSE 8005
CMD ["ragbuilder"]
