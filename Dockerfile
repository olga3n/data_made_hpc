FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    vim g++ make bc libatlas-base-dev libopenblas-dev

COPY . /opt/benchmark/

WORKDIR /opt/benchmark
RUN make
CMD make run
