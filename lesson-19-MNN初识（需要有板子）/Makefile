# Makefile
# Lesson 19: MNN C++ 推理

# MNN 头文件目录 + 库目录
MNN_HOME := $(realpath ../mnn-src)
MNN_BIN  := $(realpath ../mnn-src/build)

CXXFLAGS := -std=c++11 -I$(MNN_HOME)/include -I$(MNN_HOME)/include/MNN
LDFLAGS  := -L$(MNN_BIN) -lMNN -Wl,-rpath,$(MNN_BIN)

.PHONY: all run clean

mnn_cpp: 03_mnn_cpp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run: mnn_cpp
	./mnn_cpp

clean:
	rm -f mnn_cpp