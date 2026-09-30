# Makefile
# Lesson 11: 性能分析与调优

CC := gcc
ORT_HOME := $(realpath ../ort-bin)
CFLAGS  := -std=c17 -D_POSIX_C_SOURCE=199309L -I$(ORT_HOME)/include
LD_FLAGS := -L$(ORT_HOME)/lib -lonnxruntime -Wl,-rpath,$(ORT_HOME)/lib

.PHONY: all run clean

main: main.c
	$(CC) $(CFLAGS) -o $@ $< $(LD_FLAGS)

clean:
	rm -f main mlp3.onnx profile.json_*