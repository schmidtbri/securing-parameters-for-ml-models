# syntax=docker/dockerfile:1
FROM python:3.9-slim

ARG BUILD_DATE

LABEL org.opencontainers.image.title="Diabetes Risk Model Service"
LABEL org.opencontainers.image.description="Diabetes Risk Model Service."
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.authors="6666331+schmidtbri@users.noreply.github.com"
LABEL org.opencontainers.image.source="https://github.com/schmidtbri/securing-parameters-for-ml-models"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT License"
LABEL org.opencontainers.image.base.name="python:3.9-slim"

WORKDIR /service

ARG USERNAME=service-user
ARG USER_UID=10000
ARG USER_GID=10000

# install packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends sudo && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# create a user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# installing dependencies
COPY ./service_requirements.txt ./service_requirements.txt
RUN pip install --no-cache -r service_requirements.txt

# copying model code and license
COPY ./diabetes_risk_model ./diabetes_risk_model
COPY ./LICENSE ./LICENSE

USER $USERNAME

RUN sudo chown $USERNAME:$USERNAME -R /service && \
    sudo chmod -R +rw /service  && \
    sudo mkdir -p  /var/folders/vb && \
    sudo chown $USERNAME:$USERNAME -R /var/folders/vb && \
    sudo chmod -R +rw /var/folders/vb

CMD ["uvicorn", "rest_model_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
