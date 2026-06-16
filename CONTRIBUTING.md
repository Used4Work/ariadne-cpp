# Contributing to Ariadne

## Quick Start

```bash
# Clone and build
git clone https://github.com/Used4Work/ariadne-cpp.git
cd ariadne-cpp
sudo apt install libcurl4-openssl-dev nlohmann-json3-dev  # Linux
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j4
./build/unit_tests  # 121 tests must pass
```

## Development Guidelines

- **C++17**, namespace `ariadne`, `using json = nlohmann::json`
- All code in two files: `ariadne.hpp` (declarations) + `ariadne.cpp` (implementations)
- Comments in Chinese
- Commit format: `type(scope): description` (conventional commits)
- No raw `std::cout`/`std::cerr` in library code -- use `log_msg()`
- Thread safety: use `shared_mutex` for shared data, `atomic` for flags
- Tests: add to `tests/unit_test.cpp` using `RUN()`/`ASSERT()` macros

## Pull Request Process

1. Fork and create a feature branch from `main`
2. Implement with tests (all 121+ existing tests must still pass)
3. Run with sanitizers: `-fsanitize=address,undefined`
4. Submit PR against `main`

## Reporting Issues

Include: Ariadne version, compiler, OS, minimal reproduction code, expected vs actual behavior.
