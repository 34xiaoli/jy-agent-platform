# syntax=docker/dockerfile:1
# 审计合规 Agent 平台 · 能力服务层（对应简历中的 FastAPI 能力服务）
#
# 设计要点：
# 1. 多阶段构建：builder 装依赖，runtime 只留运行期所需，镜像更小、攻击面更小；
# 2. 非 root 运行：容器内不使用 root，符合安全基线；
# 3. 知识库与索引以 volume 挂载：镜像不含业务数据，便于合规审查与独立更新；
# 4. HEALTHCHECK 直接打 /health：容器健康 = 服务真的能处理请求，而不只是进程在。

# ---------- builder ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    LLM_PROVIDER=mock \
    TOP_K=5 \
    RECALL_K=20
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app ./app
COPY scripts ./scripts
COPY mcp ./mcp
COPY eval ./eval
COPY data/knowledge ./data/knowledge

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data/index /app/data/traces \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status==200 else 1)"

# 默认离线 mock 后端：镜像不依赖 GPU / 网络也能起来，便于演示与 CI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
