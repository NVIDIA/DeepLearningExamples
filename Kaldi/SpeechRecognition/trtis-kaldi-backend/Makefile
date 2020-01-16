.PHONY: all
all: kaldibackend

kaldibackend: kaldi-backend.cc kaldi-backend-utils.cc
	g++ -fpic -shared -std=c++11 -o libkaldi-trtisbackend.so kaldi-backend.cc kaldi-backend-utils.cc -Icustom-backend-sdk/include custom-backend-sdk/lib/libcustombackend.a -I/opt/kaldi/src/ -I/usr/local/cuda/include -I/opt/kaldi/tools/openfst/include/ -L/opt/kaldi/src/lib/ -lkaldi-cudadecoder
