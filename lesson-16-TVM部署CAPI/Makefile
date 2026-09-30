# Makefile
# Lesson 16: TVM 部署 C API

CC := gcc
TVM_HOME := $(realpath ../tvm-src)
TVM_BIN  := $(realpath ../tvm-bin)

CFLAGS  := -std=c17 -I$(TVM_HOME)/include -I$(TVM_HOME)/3rdparty/dlpack/include -I$(TVM_HOME)/3rdparty/dmlc-core/include
LD_FLAGS := -L$(TVM_BIN) -ltvm_runtime -ldl -lpthread -Wl,-rpath,$(TVM_BIN)

.PHONY: all run clean

main: main.c
	$(CC) $(CFLAGS) -o $@ $< $(LD_FLAGS)

run: main
	./main

clean:
	rm -f main