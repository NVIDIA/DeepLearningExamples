# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.09-py3

FROM ${FROM_IMAGE_NAME} AS dgl_builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y git build-essential python3-dev make cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /dgl
RUN git clone --branch v0.7.0 --recurse-submodules --depth 1 https://github.com/dmlc/dgl.git .
RUN sed -i 's/"35 50 60 70"/"60 70 80"/g' cmake/modules/CUDA.cmake
WORKDIR build
RUN cmake -DUSE_CUDA=ON -DUSE_FP16=ON ..
RUN make -j8


FROM ${FROM_IMAGE_NAME}

# VERY IMPORTANT, DO NOT REMOVE:
ENV FORCE_CUDA 1
RUN pip install -v torch-geometric
RUN pip install -v torch-scatter
RUN pip install -v torch-sparse
RUN pip install -v torch-cluster
RUN pip install -v torch-spline-conv


# copy built DGL and install it
COPY --from=dgl_builder /dgl ./dgl
RUN cd dgl/python && python setup.py install && cd ../.. && rm -rf dgl
ENV DGLBACKEND=pytorch
#RUN pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html


# HH-Suite
RUN git clone https://github.com/soedinglab/hh-suite.git && \
    mkdir -p hh-suite/build
WORKDIR hh-suite/build
RUN cmake .. && \
    make && \
    make install


# PSIPRED
WORKDIR /workspace
RUN wget http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_linux64.tar.gz -O csblast-2.2.3.tar.gz && \
mkdir -p csblast-2.2.3 && \
tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1 && \
rm csblast-2.2.3.tar.gz

RUN wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz && \
    tar xf blast-2.2.26-x64-linux.tar.gz && \
    rm blast-2.2.26-x64-linux.tar.gz

RUN wget http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/psipred.4.02.tar.gz && \
    tar xf psipred.4.02.tar.gz && \
    rm psipred.4.02.tar.gz


ADD . /workspace/rf
WORKDIR /workspace/rf

RUN wget https://openstructure.org/static/lddt-linux.zip -O lddt.zip && unzip -d lddt -j lddt.zip

RUN pip install -r requirements.txt
