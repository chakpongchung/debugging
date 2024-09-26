cmake_minimum_required(VERSION 3.24)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CUDA_GENE_FLAGS_DEBUG "-gencode arch=compute_80,code=sm_80")

add_library(ai_support SHARED)
#add_library(ai::support ALIAS ai_support)
set_target_properties(ai_support PROPERTIES EXPORT_NAME support)
set(sources
        src/flash.cu
)

target_sources(ai_support PRIVATE ${sources})
target_include_directories(ai_support PUBLIC include) # PUBLIC is needed if we need to expose it to other modules

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os;import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE OUTPUT_PYTORCH_CMAKE_PREFIX_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Pytorch cmakefile ${OUTPUT_PYTORCH_CMAKE_PREFIX_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${OUTPUT_PYTORCH_CMAKE_PREFIX_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


target_link_libraries(ai_support "${TORCH_LIBRARIES}")


set_target_properties(ai_support PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
include(GNUInstallDirs)


add_executable(example-app main.cu)
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
target_link_libraries(example-app PRIVATE ai_support)

add_executable(gemm gemm.cu)
set_property(TARGET gemm PROPERTY CXX_STANDARD 17)
target_link_libraries(gemm PRIVATE ai_support)



#install(
#        TARGETS ai_support
#        EXPORT aiTargets
#        LIBRARY
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#        COMPONENT shlib
#        ARCHIVE
#        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
#        COMPONENT lib
#        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
#)

