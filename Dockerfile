# Freqtrade remains the core trading runtime. This image only layers the
# Karakana runtime package and sidecar CLIs onto Freqtrade. Strategy files,
# sqlite databases, configs, and policies stay in the external user_data mount.
#
# Standalone build context (this repo):
#   Dockerfile
#   pyproject.toml / README.md      -> karakana wheel metadata
#   karakana/                       -> the Python package (runtime CLIs live here)
#   config/karakana-config.json     -> shipped to /freqtrade/karakana/
FROM freqtradeorg/freqtrade:2026.3 AS installer

USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get -yq update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

USER ftuser
RUN python -m pip install --upgrade pip && \
    rm -rf /home/ftuser/.local/lib/python*/site-packages/numpy* \
           /home/ftuser/.local/lib/python*/site-packages/pandas* \
           /home/ftuser/.local/lib/python*/site-packages/torch*
RUN python -m pip install --user \
    --no-cache-dir \
    termcolor \
    pygments \
    ft-scikit-optimize==0.9.2
RUN python -m pip install --user --no-cache-dir --force-reinstall \
        "torch==2.11.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --user \
    --no-cache-dir \
    --force-reinstall \
    "numpy==2.2.6" \
    "pandas==2.3.3"

COPY --chown=ftuser:ftuser pyproject.toml /tmp/karakana/pyproject.toml
COPY --chown=ftuser:ftuser README.md /tmp/karakana/README.md
COPY --chown=ftuser:ftuser karakana /tmp/karakana/karakana
COPY --chown=ftuser:ftuser config/karakana-config.json /tmp/karakana/config/karakana-config.json
RUN python -m pip install --user --no-cache-dir --no-deps --force-reinstall /tmp/karakana && \
    python -m pip install --user --no-cache-dir --force-reinstall "ccxt==4.5.54" "setuptools<82" && \
    python -m pip install --user --no-cache-dir gymnasium==1.2.3 && \
    python -c "import numpy; import pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)" && \
    python -c "from karakana.runtime.policy import KarakanaOfflinePolicy; import karakana.runtime.shadow_trader; import karakana.runtime.strategy_health" && \
    command -v karakana-shadow-trader && \
    command -v karakana-shadow-health

FROM freqtradeorg/freqtrade:2026.3
USER root
RUN DEBIAN_FRONTEND=noninteractive apt-get -yq update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/freqtrade/

USER ftuser
COPY --from=installer --chown=ftuser:ftuser /home/ftuser/.local /home/ftuser/.local

RUN mkdir -p /freqtrade/karakana /freqtrade/plot_equity /freqtrade/user_data/logs
COPY --chown=ftuser:ftuser config/karakana-config.json /freqtrade/karakana/
RUN touch /freqtrade/user_data/__init__.py /freqtrade/user_data/logs/freqtrade.log
RUN python -c "from karakana.runtime.policy import KarakanaOfflinePolicy; import karakana.runtime.shadow_trader; import karakana.runtime.strategy_health" && \
    command -v karakana-shadow-trader && \
    command -v karakana-shadow-health

USER ftuser
