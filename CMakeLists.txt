cmake_minimum_required(VERSION 3.15)
project(FashionVisionAI CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O2 -Wall)
endif()

add_library(fv_core
    src/tensor.cpp
    src/linalg.cpp
    src/gradcheck.cpp
    src/viz.cpp
    src/neuron.cpp
    src/mlp.cpp
    src/dataset.cpp
    src/optimizer.cpp
    src/trainer.cpp)
target_include_directories(fv_core PUBLIC include)

add_executable(fashionvision src/main.cpp)
target_link_libraries(fashionvision fv_core)

# foreach(d tensor_demo mlp_demo optim_demo init_demo)
foreach(d tensor_demo mlp_demo optim_demo init_demo norm_demo reg_demo augment_demo overfit_demo learning_rules_demo boltzmann_demo)
add_executable(${d} src/${d}.cpp)
    target_link_libraries(${d} fv_core)
endforeach()

enable_testing()
foreach(t test_tensor test_linalg test_mlp test_optim)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} fv_core)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
# ONNX Runtime C++ Integration
set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
)

add_executable(resnet_onnx
    src/resnet_onnx.cpp
)

target_include_directories(resnet_onnx PRIVATE
    "${ONNXRUNTIME_ROOT}/build/native/include"
)

target_link_libraries(resnet_onnx PRIVATE
    "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.lib"
)

add_custom_command(TARGET resnet_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.dll"
    "$<TARGET_FILE_DIR:resnet_onnx>"
)
# C++ ONNX Performance Benchmark
add_executable(benchmark_onnx
    src/benchmark_onnx.cpp
)

target_include_directories(benchmark_onnx PRIVATE
    "${ONNXRUNTIME_ROOT}/build/native/include"
)

target_link_libraries(benchmark_onnx PRIVATE
    "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.lib"
)

add_custom_command(TARGET benchmark_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.dll"
    "$<TARGET_FILE_DIR:benchmark_onnx>"
)