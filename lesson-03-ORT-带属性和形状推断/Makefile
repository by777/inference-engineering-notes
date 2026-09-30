# Makefile — Lesson 03
CXX := g++
CXXFLAGS := -std=c++17 -I../ort-bin/include -fPIC
TARGET := libmy_ops.so
SRC := my_ops.cc

.PHONY: all gen run clean


all: $(TARGET)
$(TARGET): $(SRC)
	$(CXX) $(SRC) -shared -o $(TARGET) $(CXXFLAGS)

gen:
	python3 gen_model.py

run: gen all
	python3 test_model.py
clean:
	rm -f $(TARGET) my_add.onnx