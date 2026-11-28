FROM public.ecr.aws/lambda/python:3.12

RUN dnf install -y libgomp && dnf clean all

COPY lambdas/predict/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY src/fdf ${LAMBDA_TASK_ROOT}/fdf
COPY config/airports.csv ${LAMBDA_TASK_ROOT}/config/airports.csv
COPY models/2026-11-27-lgbm-v2 ${LAMBDA_TASK_ROOT}/model
COPY lambdas/predict/handler.py ${LAMBDA_TASK_ROOT}/handler.py

ENV MODEL_DIR=${LAMBDA_TASK_ROOT}/model
ENV AIRPORTS_CSV=${LAMBDA_TASK_ROOT}/config/airports.csv

CMD ["handler.lambda_handler"]
