cmake_minimum_required(VERSION 3.20)
project(edialect)

# ===== 路径 =====
set(MLIR_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-src/build")
set(MLIR_SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-src/mlir")
set(LLVM_SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../../mlir-src/llvm")
# 生成的 .inc 放 build/gen/Toy/（带 Toy/ 子目录，匹配源码里
# include "Toy/xxx.h.inc" 的写法；include path 只加 gen 父目录）
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen/Toy")

# ===== include 路径（4 个来源） =====
include_directories(
  # 1. 我们自己的源码头文件（Toy/ToyDialect.h 等）
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  # 2. LLVM 源码头文件
  ${LLVM_SRC_DIR}/include
  # 3. MLIR/LLVM 生成头文件（mlir/IR/Dialect.h 等在 build 里生成）
  ${MLIR_BUILD_DIR}/include
  ${MLIR_BUILD_DIR}/tools/mlir/include
  # 4. MLIR 源码头文件
  ${MLIR_SRC_DIR}/include
  # 5. 我们的生成目录（.inc）。只加 gen 父目录，
  #    这样源码里 include "Toy/ToyOps.h.inc" 能解析到
  ${CMAKE_CURRENT_BINARY_DIR}/gen
)

# ===== 关键：调用 mlir-tblgen 生成 4 个 .inc =====
# add_custom_command：声明"某产物由某命令生成"，ninja 自动管理依赖
#   每次改 .td 重新 make，.inc 自动重新生成（增量）
set(TBLGEN ${MLIR_BUILD_DIR}/bin/mlir-tblgen)
# tblgen 的 include 路径：它要解析 .td 里 include 的
# "mlir/IR/OpBase.td"（在 MLIR 源码）和 "Toy/ToyDialect.td"（在我们 include/）
set(TD_INCLUDES
  -I ${MLIR_SRC_DIR}/include
  -I ${MLIR_BUILD_DIR}/tools/mlir/include
  -I ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(MAKE_DIRECTORY ${GEN_DIR})

# 1. 方言类声明：ToyDialect.h.inc
add_custom_command(
  OUTPUT ${GEN_DIR}/ToyDialect.h.inc
  COMMAND ${TBLGEN} -gen-dialect-decls ${TD_INCLUDES}
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyDialect.td
          -o ${GEN_DIR}/ToyDialect.h.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyDialect.td)

# 2. 方言类实现：ToyDialect.cpp.inc
add_custom_command(
  OUTPUT ${GEN_DIR}/ToyDialect.cpp.inc
  COMMAND ${TBLGEN} -gen-dialect-defs ${TD_INCLUDES}
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyDialect.td
          -o ${GEN_DIR}/ToyDialect.cpp.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyDialect.td)

# 3. op 类声明：ToyOps.h.inc
add_custom_command(
  OUTPUT ${GEN_DIR}/ToyOps.h.inc
  COMMAND ${TBLGEN} -gen-op-decls ${TD_INCLUDES}
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyOps.td
          -o ${GEN_DIR}/ToyOps.h.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyOps.td)

# 4. op 类实现：ToyOps.cpp.inc
add_custom_command(
  OUTPUT ${GEN_DIR}/ToyOps.cpp.inc
  COMMAND ${TBLGEN} -gen-op-defs ${TD_INCLUDES}
          ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyOps.td
          -o ${GEN_DIR}/ToyOps.cpp.inc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/Toy/ToyOps.td)

# ===== 生成依赖：保证 4 个 .inc 先于编译生成 =====
# 踩坑：只把 .h.inc 列进 add_executable 的源，ninja 只生成被依赖的 2 个，
# .cpp.inc 从不生成。必须建 custom target 依赖全部 4 个，
# 再用 add_dependencies 挂到可执行文件上。
add_custom_target(toy-inc DEPENDS
  ${GEN_DIR}/ToyDialect.h.inc
  ${GEN_DIR}/ToyDialect.cpp.inc
  ${GEN_DIR}/ToyOps.h.inc
  ${GEN_DIR}/ToyOps.cpp.inc)

# ===== 可执行文件 =====
add_executable(edialect-opt
  tools/edialect-opt.cpp
  lib/Toy/ToyDialect.cpp
  lib/Toy/ToyOps.cpp)

# -fno-rtti：MLIR 库编译时关了 RTTI，我们的代码必须一致（同 D 组）
set_target_properties(edialect-opt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  COMPILE_FLAGS "-fno-rtti")

# 关键依赖：先跑 tblgen 生成 .inc，再编译 .cpp
add_dependencies(edialect-opt toy-inc)

# 全量链接 MLIR + LLVM 静态库（同 D 组方案）
# --start-group：解决静态库循环依赖
file(GLOB MLIR_LIBS ${MLIR_BUILD_DIR}/lib/libMLIR*.a)
file(GLOB LLVM_LIBS ${MLIR_BUILD_DIR}/lib/libLLVM*.a)
target_link_libraries(edialect-opt
  -Wl,--start-group ${MLIR_LIBS} ${LLVM_LIBS} -Wl,--end-group
  pthread z dl)
