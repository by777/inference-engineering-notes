# Lesson 20 Makefile: 交叉编译 02_main_arm + 推到板子跑
#
# 用法:
#   make            # 交叉编译 02_main_arm
#   make run        # 推送 + 板子上跑 4 个版本
#   make clean      # 清产物
#
# 环境变量（可覆盖）:
#   NDK=/path/to/ndk     (默认 r27d)
#   API=33               (Android API level)

NDK ?= $HOME/toolchain/android-ndk-r27d
API ?= 33
TVMSRC := $(abspath $(CURDIR)/../tvm-src)

CXX := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android$(API)-clang++
OBJDUMP := aarch64-linux-gnu-objdump
ADB := adb

CFLAGS := -O3 -fPIE -pie -fexceptions
INCLUDES := \
  -I$(TVMSRC)/include \
  -I$(TVMSRC)/3rdparty/dlpack/include \
  -I$(TVMSRC)/3rdparty/dmlc-core/include
LDLIBS := -L. -ltvm_runtime -Wl,-rpath,.

BOARD_DIR := /data/local/tmp/lesson20
LIBCXX_SHARED := $(NDK)/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
TVM_RT := $(abspath $(CURDIR)/../tvm-bin-arm/libtvm_runtime.so)

.PHONY: all build-cpp clean push run

all: build-cpp

build-cpp: 02_main_arm

# @： 表示不显示命令本身，只显示输出结果
# file： 检测文件类型（ELF 架构、平台）
# @: Make 自动化变量 = 当前规则的目标文件名
# sed 's/^/  /' = 在每一行行首前加两个空格（^ 是行首锚点）
02_main_arm: 02_main_arm.cpp
	$(CXX) $(CFLAGS) $(INCLUDES) $< $(LDLIBS) -o $@
	@file $@ | sed 's/^/  /'

# 推到板子 + 跑 4 个版本
push:
	$(ADB) shell mkdir -p $(BOARD_DIR)
	$(ADB) push libmatmul_naive.so        $(BOARD_DIR)/
	$(ADB) push libmatmul_opt.so          $(BOARD_DIR)/
	$(ADB) push libmatmul_parallel.so     $(BOARD_DIR)/
	$(ADB) push libmatmul_opt_parallel.so $(BOARD_DIR)/
	$(ADB) push 02_main_arm               $(BOARD_DIR)/
	$(ADB) push $(TVM_RT)                 $(BOARD_DIR)/libtvm_runtime.so
	$(ADB) push $(LIBCXX_SHARED)          $(BOARD_DIR)/libc++_shared.so
	$(ADB) shell chmod +x $(BOARD_DIR)/02_main_arm

# $$so → 关键点：Make 会先把 $ 当成自己的变量符号，
# $$ 转义后传给 shell 就变成 $so（shell 循环变量）。对比 $(ADB)、$(BOARD_DIR) 是 Make 变量，在 Make 解析阶段就被替换了
run: push
	@#recipe 里（以 Tab 缩进的命令行）— # 会被传给 shell ⚠️
	@#以 Tab 开头的行是 shell 命令，Make 不会把 # 当注释，而是整行传给 sh。碰巧 shell 也认 # 注释，所以直接写 # xxx 通常也能工作——但这是 shell 的注释，不是 Make 的。
	@#常见的规范写法是 @# xxx：
	@for so in libmatmul_naive.so libmatmul_opt.so libmatmul_parallel.so libmatmul_opt_parallel.so; do \
	  echo "------ $$so ------"; \
	  $(ADB) shell "cd $(BOARD_DIR) && LD_LIBRARY_PATH=. ./02_main_arm $$so 20"; \
	  echo; \
	done

clean:
	rm -f 02_main_arm libmatmul_*.so *.disasm *_neon_only.txt