# Makefile — Lesson 01
# 用法: make        编译
#      make run     编译并运行
#      make gen     生成 ONNX 模型
#      make clean   清理

CXX := g++
CXXFLAGS := -std=c++17 -I../ort-bin/include
LDFLAGS := -L../ort-bin/lib -lonnxruntime -Wl,-rpath,../ort-bin/lib

TARGET := step1
SRC := step1_myadd.cpp

.PHONY: all run gen clean

all: $(TARGET)
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

gen:
	python3 gen_model.py
run: gen $(TARGET)
	./$(TARGET)
clean:
	rm -f $(TARGET) *.onnx