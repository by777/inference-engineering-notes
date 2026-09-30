# Makefile
# Lesson 08: C API 推理

CC := gcc
# $(realpath ../ort-bin)表示把 ../ort-bin 解析成绝对路径
# 用 realpath 的好处：即使你在别的目录调用 make，路径也不会乱
ORT_HOME := $(realpath ../ort-bin)

CFLAGS  := -std=c17 -I$(ORT_HOME)/include
# -L$(ORT_HOME)/lib	-L = Library 搜索路径（找 .so 时用）
# -lonnxruntime	-l = link，链接 libonnxruntime.so（去掉 lib 和 .so
# -Wl,-rpath,...	把库路径写进可执行文件，运行时自动去那里找 .so
# -L         只在编译链接时用           ld: 找不到库就报错
# -rpath     写入可执行文件，运行时用   运行时动态加载器去找
# 没有 -rpath 的话，运行时你得手动设置：export LD_LIBRARY_PATH=$HOME/.../ort-bin/lib
LD_FLAGS := -L$(ORT_HOME)/lib -lonnxruntime -Wl,-rpath,$(ORT_HOME)/lib

.PHONY: all run clean
# $@ 长得像靶心，表示冒号左边的目标，也是main
# $< 表示冒号右边的第一个依赖文件，也就是main.c
# $^ 向上指 ↕, 表示冒号右边的所有依赖文件
# 其它还有：
# %.o: %.c # 表示把所有.c 文件编译成 .o 文件
# main: a.o b.o c.o # 表示把 a.o b.o c.o 链接成 main 可执行文件
main: main.c
	$(CC) $(CFLAGS)  -o $@ $< $(LD_FLAGS)
	

run: main
	./main

clean:
	rm -f main