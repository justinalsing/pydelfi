# This container builds the Bifrost documentation
# using the ledatelescope/bifrost container, and
# puts it into the folder /bifrost/docs/build/html
# inside a new container.
FROM tensorflow/tensorflow:latest-py3

# Install pre-requisite documentation libraries at
# a specific commit that is known to work
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential \
        curl \
        git \
        pkg-config \
        software-properties-common \
        doxygen \
        exuberant-ctags \
        nano \
        vim \
        && \
    pip3 install --no-cache-dir \
    git+https://github.com/sphinx-doc/sphinx.git \
    git+https://github.com/michaeljones/breathe.git

WORKDIR "/pydelfi"

# pydelfi install
RUN git clone https://github.com/milescranmer/pydelfi /pydelfi && \
    ls && \
    cd /pydelfi && \
    python setup.py install

# Build the docs
RUN cd /pydelfi && \
    ls && \
    cat Makefile && \ 
    make doc && \
    cd docs && \
    make html

WORKDIR "/pydelfi/docs/build/html"

RUN ["/bin/bash"]
