cmake_minimum_required(VERSION 3.9)
project(gtda_bindings LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/giotto_ph/pybind11)
set(BASE_LIB_DIR "giotto_ph")

set(BINDINGS_DIR "${BASE_LIB_DIR}/bindings")
set(EXT_DIR "${BASE_LIB_DIR}/ext")

# Base folder containing C++ Backend
set(CPP_SRC_DIR "${BASE_LIB_DIR}/src")
set(CPP_INC_DIR "${BASE_LIB_DIR}/inc")


#######################################################################
#                              Collapser                              #
#######################################################################

pybind11_add_module(gtda_collapser MODULE "${BINDINGS_DIR}/collapser_bindings.cpp")
set_property(TARGET gtda_collapser PROPERTY CXX_STANDARD 14)

if(OpenMP_FOUND)
    target_link_libraries(gtda_collapser PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(gtda_collapser PRIVATE "${CPP_SRC_DIR}/")

if(MSVC)
    target_compile_options(gtda_collapser PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gtda_collapser PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gtda_collapser PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fPIC -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gtda_collapser PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()
