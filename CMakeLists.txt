cmake_minimum_required(VERSION 3.9)
project(gtda_bindings LANGUAGES CXX)

set(BASE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gph)
set(BINDINGS_DIR ${BASE_LIB_DIR}/bindings)
set(EXT_DIR ${BASE_LIB_DIR}/ext)

# Base folder containing C++ Backend
set(CPP_SRC_DIR ${BASE_LIB_DIR}/src)
set(CPP_INC_DIR ${BASE_LIB_DIR}/inc)

add_subdirectory(${EXT_DIR}/pybind11)

# The following option is require to enable on all targets -fPIC
# Or similar for all compilers
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# See details of this requirements
# https://crascit.com/2015/03/28/enabling-cxx11-in-cmake/
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TURF_PREFER_CPP11 TRUE CACHE BOOL "Prefer C++11 atomics, threads, etc. over native implementations")
set(JUNCTION_WITH_SAMPLES FALSE CACHE BOOL "Include all Junction samples in generated build system")
add_subdirectory(${EXT_DIR}/junction)

#######################################################################
#                          Ripser Lock Free                           #
#######################################################################

pybind11_add_module(gph_ripser MODULE "${BINDINGS_DIR}/ripser_bindings.cpp")
set_property(TARGET gph_ripser PROPERTY CXX_STANDARD 14)

if(OpenMP_FOUND)
  target_link_libraries(gph_ripser PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(gph_ripser PRIVATE ${CPP_SRC_DIR})
target_include_directories(gph_ripser PRIVATE ${CPP_INC_DIR})
target_include_directories(gph_ripser PRIVATE ${JUNCTION_ALL_INCLUDE_DIRS})

if(MSVC)
    target_compile_options(gph_ripser PUBLIC $<$<CONFIG:RELEASE>: /Wall /O2>)
    target_compile_options(gph_ripser PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gph_ripser PUBLIC $<$<CONFIG:RELEASE>: -O3 -g>)
    target_compile_options(gph_ripser PUBLIC $<$<CONFIG:DEBUG>: -O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

target_link_libraries(gph_ripser PRIVATE ${JUNCTION_ALL_LIBRARIES})

#######################################################################
#                Ripser Lock Free - Coefficient enable                #
#######################################################################

pybind11_add_module(gph_ripser_coeff MODULE "${BINDINGS_DIR}/ripser_bindings.cpp")
set_property(TARGET gph_ripser_coeff PROPERTY CXX_STANDARD 14)

if(OpenMP_FOUND)
  target_link_libraries(gph_ripser_coeff PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(gph_ripser_coeff PRIVATE USE_COEFFICIENTS=1)
target_include_directories(gph_ripser_coeff PRIVATE ${CPP_SRC_DIR})
target_include_directories(gph_ripser_coeff PRIVATE ${CPP_INC_DIR})
target_include_directories(gph_ripser_coeff PRIVATE ${JUNCTION_ALL_INCLUDE_DIRS})

if(MSVC)
    target_compile_options(gph_ripser_coeff PUBLIC $<$<CONFIG:RELEASE>: /Wall /O2>)
    target_compile_options(gph_ripser_coeff PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gph_ripser_coeff PUBLIC $<$<CONFIG:RELEASE>: -O3>)
    target_compile_options(gph_ripser_coeff PUBLIC $<$<CONFIG:DEBUG>: -O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()

target_link_libraries(gph_ripser_coeff PRIVATE ${JUNCTION_ALL_LIBRARIES})

#######################################################################
#                              Collapser                              #
#######################################################################

pybind11_add_module(gph_collapser MODULE "${BINDINGS_DIR}/collapser_bindings.cpp")
set_property(TARGET gph_collapser PROPERTY CXX_STANDARD 14)

if(OpenMP_FOUND)
    target_link_libraries(gph_collapser PRIVATE OpenMP::OpenMP_CXX)
endif()

target_include_directories(gph_collapser PRIVATE "${CPP_SRC_DIR}/")

if(MSVC)
    target_compile_options(gph_collapser PUBLIC $<$<CONFIG:RELEASE>: /O2 /Wall /fp:strict>)
    target_compile_options(gph_collapser PUBLIC $<$<CONFIG:DEBUG>:/O1 /DEBUG:FULL /Zi /Zo>)
else()
    target_compile_options(gph_collapser PUBLIC $<$<CONFIG:RELEASE>: -Ofast -shared -pthread -fwrapv -Wall -fno-strict-aliasing -frounding-math>)
    target_compile_options(gph_collapser PUBLIC $<$<CONFIG:DEBUG>:-O2 -ggdb -D_GLIBCXX_DEBUG>)
endif()
