cmake_minimum_required(VERSION 3.17)
project(ariadne VERSION 0.2.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CURL REQUIRED)
find_package(nlohmann_json 3.9 REQUIRED)

option(ARIADNE_BUILD_TESTS    "Build unit tests"  ON)
option(ARIADNE_BUILD_EXAMPLES "Build examples"    ON)

add_compile_options(-Wall -Wextra)

# ── Core library ──────────────────────────────────────────────────
add_library(ariadne STATIC ariadne.cpp)
add_library(ariadne::ariadne ALIAS ariadne)
target_include_directories(ariadne PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
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
