FROM tensorflow/tensorflow:2.0.0a0

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

COPY ./flights.py /flights.py
COPY ./data/*.* /data/

ENTRYPOINT [ "python","flights.py" ]