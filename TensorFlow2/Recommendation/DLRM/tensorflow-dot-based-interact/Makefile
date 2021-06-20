# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

.DEFAULT_GOAL := lib

CC_SRC_DIR = tensorflow_dot_based_interact/cc
CC_PY_DIR = tensorflow_dot_based_interact/python
CC_SRCS = $(CC_SRC_DIR)/kernels/dot_based_interact_kernels.cc \
          $(CC_SRC_DIR)/kernels/dot_based_interact_grad_kernels.cc \
          $(CC_SRC_DIR)/ops/dot_based_interact_ops.cc
VOLTA_TARGET_OBJECT = $(CC_SRC_DIR)/_dot_based_interact_volta.cu.o
AMPERE_TARGET_OBJECT = $(CC_SRC_DIR)/_dot_based_interact_ampere.cu.o
TARGET_LIB = $(CC_PY_DIR)/ops/_dot_based_interact_ops.so

volta: $(VOLTA_TARGET_OBJECT)
$(VOLTA_TARGET_OBJECT): $(CC_SRC_DIR)/kernels/volta/dot_based_interact_volta.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr -arch=sm_70

ampere: $(AMPERE_TARGET_OBJECT)
$(AMPERE_TARGET_OBJECT): $(CC_SRC_DIR)/kernels/ampere/dot_based_interact_ampere.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr -arch=sm_80

lib: $(TARGET_LIB)
$(TARGET_LIB): $(CC_SRCS) $(VOLTA_TARGET_OBJECT) $(AMPERE_TARGET_OBJECT)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

test: $(CC_PY_DIR)/ops/dot_based_interact_ops_test.py $(CC_PY_DIR)/ops/dot_based_interact_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) $(CC_PY_DIR)/ops/dot_based_interact_ops_test.py

pkg: $(TARGET_LIB)
	./build_pip_pkg.sh

clean:
	rm -f $(VOLTA_TARGET_OBJECT) $(TARGET_LIB)
	rm -rf artifacts
