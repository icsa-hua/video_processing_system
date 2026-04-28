
# `python-base` sets up all our shared environment variables
# FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3  as python-base

#     # python
# ENV PYTHONUNBUFFERED=1 \
#     # prevents python creating .pyc files
#     PYTHONDONTWRITEBYTECODE=1 \
#     \
#     # pip
#     PIP_NO_CACHE_DIR=off \
#     PIP_DISABLE_PIP_VERSION_CHECK=on \
#     PIP_DEFAULT_TIMEOUT=100 \
#     \
#     # poetry
#     # https://python-poetry.org/docs/configuration/#using-environment-variables
#     # make poetry install to this location
#     POETRY_HOME="/opt/poetry" \
#     # make poetry create the virtual environment in the project's root
#     # it gets named `.venv`
#     POETRY_VIRTUALENVS_IN_PROJECT=true \
#     # do not ask any interactive question
#     POETRY_NO_INTERACTION=1 \
#     \
#     # paths
#     # this is where our requirements + virtual environment will live
#     PYSETUP_PATH="/opt/pysetup" \
#     VENV_PATH="/opt/pysetup/.venv"


# # prepend poetry and venv to path
# ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# # `builder-base` stage is used to build deps + create our virtual environment
# FROM python-base as builder-base
# RUN apt-get update \
#     && apt-get install --no-install-recommends -y \
#         # deps for installing poetry
#         curl \
#         # deps for building python deps
#         build-essential

# # install poetry - respects $POETRY_VERSION & $POETRY_HOME
# RUN curl -sSL https://install.python-poetry.org | python3 -

# # copy project requirement files here to ensure they will be cached.
# WORKDIR $PYSETUP_PATH
# COPY poetry.lock pyproject.toml ./

# # install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
# RUN poetry install --no-directory


# # `development` image is used during development / testing
# FROM python-base as development
# ENV FASTAPI_ENV=development
# WORKDIR $PYSETUP_PATH

# # copy in our built poetry + venv
# COPY --from=builder-base $POETRY_HOME $POETRY_HOME
# COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

# # quicker install as runtime deps are already installed
# RUN poetry install

# FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
# FROM dustynv/l4t-pytorch:r36.4.0
#
# # MAke sure no prompts stop the installations
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Etc/UTC
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
#
# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     ffmpeg libsm6 libxext6 libxrender-dev \
#     libgl1-mesa-glx python3-pip \
#     git wget unzip \
#     python3-opencv \
#     && apt-get clean
#     
# # Install necessary packages
# RUN apt-get update 
#
# COPY README.md README.md
# COPY requirements.txt requirements.txt
# COPY setup.py setup.py
#
# RUN pip3 install -e . 
#
# WORKDIR /app 
# COPY . /app 
#
# CMD ["python3", "/app/scripts/obs_pipeline.py", "--save", "--use_TRT", "--only_FPS"]

# FROM dustynv/l4t-ml:r36.2.0 
FROM ultralytics/ultralytics:latest-jetson-jetpack6

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV_PATH=/opt/venv

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \ 
    openssl \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3-venv \
    python3-wheel \ 
    python3-setuptools \ 
    && update-ca-certificates \ 
    && rm -rf /var/lib/apt/lists/*


ENV PATH="${VENV_PATH}/bin:${PATH}"

WORKDIR /workspace

COPY requirements.txt . 

COPY setup.py . 

RUN python3 -m pip install --no-cache-dir --no-deps -r requirements.txt

RUN python3 -m pip install -e . --no-deps

CMD ["bash"]




























