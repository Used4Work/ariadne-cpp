cmake_minimum_required(VERSION 3.17)
project(ariadne VERSION 2.10.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate version header from CMake (single source of truth)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/ariadne_version_gen.hpp @ONLY)

find_package(CURL REQUIRED)
find_package(nlohmann_json 3.9 REQUIRED)

# When included via add_subdirectory/FetchContent, disable extras by default
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(_ARIADNE_DEFAULT ON)
else()
    set(_ARIADNE_DEFAULT OFF)
endif()
option(ARIADNE_BUILD_TESTS    "Build unit tests"  ${_ARIADNE_DEFAULT})
option(ARIADNE_BUILD_EXAMPLES "Build examples"    ${_ARIADNE_DEFAULT})

# Compiler warnings — MSVC uses different flags than GCC/Clang
if(MSVC)
    add_compile_options(/W3 /wd4068 /wd4702 /utf-8)  # W3 moderate, silence pragma/unreachable warnings, UTF-8 source
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ── Core library ──────────────────────────────────────────────────
add_library(ariadne STATIC ariadne.cpp)
add_library(ariadne::ariadne ALIAS ariadne)
target_include_directories(ariadne PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(ariadne PUBLIC CURL::libcurl nlohmann_json::nlohmann_json)

# ── Optional targets ──────────────────────────────────────────────
if(ARIADNE_BUILD_EXAMPLES)
    add_executable(example example.cpp)
    target_link_libraries(example PRIVATE ariadne)
endif()

if(ARIADNE_BUILD_TESTS)
    enable_testing()
    add_executable(unit_tests tests/unit_test.cpp)
    target_link_libraries(unit_tests PRIVATE ariadne)
    add_test(NAME unit_tests COMMAND unit_tests)

    add_executable(eval_runner tests/eval_runner.cpp)
    target_link_libraries(eval_runner PRIVATE ariadne)

    add_executable(benchmark tests/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE ariadne)
endif()

# ── Ariadne Studio (visual workflow editor) ──────────────────────
option(ARIADNE_BUILD_STUDIO "Build Ariadne Studio" ${_ARIADNE_DEFAULT})
if(ARIADNE_BUILD_STUDIO)
    # Auto-download cpp-httplib if not present
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/httplib.h")
        message(STATUS "Downloading cpp-httplib...")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.20.0/httplib.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/httplib.h"
            STATUS dl_status)
        list(GET dl_status 0 dl_code)
        if(NOT dl_code EQUAL 0)
            message(WARNING "Failed to download httplib.h — Studio will not build")
            set(ARIADNE_BUILD_STUDIO OFF)
        endif()
    endif()
endif()
if(ARIADNE_BUILD_STUDIO)
    # Embed index.html into a C++ header at build time
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/studio_embedded.h
        COMMAND ${CMAKE_COMMAND}
            -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/studio/index.html
            -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/studio_embedded.h
            -DVAR=studio_html
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_resource.cmake
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/studio/index.html
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_resource.cmake
        COMMENT "Embedding studio/index.html into binary"
    )
    add_custom_target(embed_studio DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/studio_embedded.h)

    add_executable(ariadne-studio studio/studio.cpp)
    add_dependencies(ariadne-studio embed_studio)
    target_include_directories(ariadne-studio PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/deps
        ${CMAKE_CURRENT_SOURCE_DIR}/studio
        ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(ariadne-studio PRIVATE ariadne)
    if(WIN32)
        target_link_libraries(ariadne-studio PRIVATE ws2_32)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
        target_link_libraries(ariadne-studio PRIVATE stdc++fs)
    endif()
endif()

# ── Install (for vcpkg / system install) ─────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ariadne EXPORT ariadneTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ariadne.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ariadneTargets
    FILE ariadneTargets.cmake
    NAMESPACE ariadne::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ariadne)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ariadneConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ariadneConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ariadne)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ariadneConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ariadne)

# pkg-config for Linux consumers
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ariadne.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/ariadne.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ariadne.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Install models.json if present (runtime dependency for model-check)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models.json)
    install(FILES models.json DESTINATION ${CMAKE_INSTALL_DATADIR}/ariadne)
endif()
