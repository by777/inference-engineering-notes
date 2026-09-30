CXX := g++
CXXFLAGS := -std=c++17 -I../ort-bin/include -fPIC
TARGET := libmy_ops.so
SRC := my_ops.cc

# 动作目标（phony target）
.PHONY: all gen run clean

# all 这个目标依赖 libmy_ops.so
# 执行 make all 时，会先确保 libmy_ops.so 存在且是最新
all: $(TARGET)
# libmy_ops.so 依赖 my_ops.cc
$(TARGET): $(SRC)

	$(CXX) $(SRC) -shared -o $(TARGET) $(CXXFLAGS)

gen:
	python3 gen_model.py axis0_keep0
	python3 gen_model.py axis1_keep0
	python3 gen_model.py axis1_keep1

# 意味着先执行/更新 gen 和 all，然后再跑 python3 test_model.py
run: gen all
	python3 test_model.py

clean:
	rm -f $(TARGET) my_reduce_mean_*.onnx