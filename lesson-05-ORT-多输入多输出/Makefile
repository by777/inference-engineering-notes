# Makefile — Lesson 05
CXX := g++
CXXFLAGS := -std=c++17 -I../ort-bin/include -fPIC
TARGET := libmy_ops.so
SRC := my_ops.cc

.PHONY: run clean

run: $(TARGET) my_add_sub.onnx
	python3 test_model.py

$(TARGET): $(SRC)
	$(CXX) $(SRC) -shared -o $(TARGET) $(CXXFLAGS)

my_add_sub.onnx: gen_model.py
	python3 gen_model.py

clean:
	rm -f $(TARGET) my_add_sub.onnx