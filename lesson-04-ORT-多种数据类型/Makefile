# Makefile — Lesson 04
CXX := g++
CXXFLAGS := -std=c++17 -I../ort-bin/include -fPIC
TARGET := libmy_ops.so
SRC := my_ops.cc

.PHONY: run clean

run: libmy_ops.so models
	python3 test_model.py

libmy_ops.so: my_ops.cc
	g++ my_ops.cc -shared -o libmy_ops.so -std=c++17 -I../ort-bin/include -fPIC

models:
	python3 gen_model.py float
	python3 gen_model.py double
	python3 gen_model.py int32
	python3 gen_model.py int64

clean:
	rm -f libmy_ops.so my_add_*.onnx