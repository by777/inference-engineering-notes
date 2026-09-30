# Makefile
# Lesson 10: Session 保存/加载

CC := gcc
ORT_HOME := $(realpath ../ort-bin)
CFLAGS  := -std=c17 -I$(ORT_HOME)/include
LD_FLAGS := -L$(ORT_HOME)/lib -lonnxruntime -Wl,-rpath,$(ORT_HOME)/lib

.PHONY: all run clean

main: main.c
	$(CC) $(CFLAGS) -o $@ $< $(LD_FLAGS)

run: main
	./main

clean:
	rm -f main original.onnx optimized.onnx optimized.ort