# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ARCH = "sm_70"  # Volta
# ARCH = "sm_75"  # Turing
# ARCH = "sm_80"  # Ampere

CUDA_PATH  =/usr/local/cuda

CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_COM_PATH = $(CUDA_PATH)/samples/common/inc

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -w -std=c++11 -DNDEBUG
INCLUDES := -I$(CUDA_COM_PATH) -I$(CUDA_INC_PATH) -I/usr/include

CUDA_LIB_PATH  = $(CUDA_PATH)/lib64
LDFLAGS := -L$(CUDA_LIB_PATH) 
LDFLAGS += -lnvinfer -lcudart

LDFLAGS += -Wl,-rpath=$(CUDA_LIB_PATH)

SO = AddPosEncPlugin.so
OBJ = $(shell find . -name '*.o')
DEP = $(OBJ:.o=.d)

all: $(SO) $(CUDA_BIN)

AddPosEncPlugin.so: AddPosEncPlugin.o

-include $(DEP)

clean:
	rm -rf $(SO) $(CUDA_BIN) $(OBJ) $(DEP)

%.o: %.cpp
	$(GCC) $(CCFLAGS) -fPIC -MD -MP $(INCLUDES) -o $@ -c $<

%.o: %.cu
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC -arch=$(ARCH) -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)

test: all
	python3 test_add_pos_enc_plugin.py
