FROM python:3.12-slim

ARG UID=1000
ARG GID=1000
ARG USERNAME=appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${GID}" "${USERNAME}" \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && chown -R "${UID}:${GID}" /app /home/"${USERNAME}"

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN chown -R "${UID}:${GID}" /app

USER ${USERNAME}

ENTRYPOINT [ "python" ]
