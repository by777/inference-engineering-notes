cmake_minimum_required(VERSION 3.20)
project(dpass)

set(MLIR_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-src/build")
set(MLIR_SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-src/mlir")
set(LLVM_SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-src/llvm")

include_directories(
  ${LLVM_SRC_DIR}/include
  ${MLIR_BUILD_DIR}/include
  ${MLIR_BUILD_DIR}/tools/mlir/include
  ${MLIR_SRC_DIR}/include
)

# 自己的mlir-opt（链接全部MLIR+LLVM静态库）
add_executable(dpass-opt dpass-opt.cpp Passes.cpp)

# 关键：MLIR库时-fno-rtti编译的，必须一致
set_target_properties(
  dpass-opt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  COMPILE_FLAGS "-fno-rtti" #不要生成运行时类型信息，
  # 1. 会导致dynamic_cast/typeid 不能用（编译报错或行为异常） 2. 程序更小、更快
)

# 全量链接（--start-group 处理循环依赖）
file(GLOB MLIR_LIBS ${MLIR_BUILD_DIR}/lib/libMLIR*.a)
file(GLOB LLVM_LIBS ${MLIR_BUILD_DIR}/lib/libLLVM*.a)

target_link_libraries(dpass-opt
  -Wl,--start-group ${MLIR_LIBS} ${LLVM_LIBS} -Wl,--end-group
  pthread z dl)

# dpass-opt——给 dpass-opt 这个可执行文件链接这些库。
# -Wl,--start-group ... -Wl,--end-group——处理循环依赖
# 静态库的链接是"单向"的——库 A 只能引用它之前已经链接的库的符号。
# g++ main.o libA.a libB.a   # 只能 A 用 B 的，B 不能用 A 的
# --start-group 解法 -Wl,--start-group libA.a libB.a libC.a -Wl,--end-group
# 告诉链接器"这组库可以反复扫描"——像转圈一样来回找符号，直到所有符号都解决或确认找不到。这样循环依赖就解决了
# pthread	POSIX 线程	MLIR 并行优化用线程（-lpthread）
# z	zlib 压缩	LLVM 的压缩支持（debug info 压缩等）
# dl	动态加载（dlopen）	MLIR 的插件加载、共享库支持