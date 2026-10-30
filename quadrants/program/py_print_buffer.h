#pragma once

#include <mutex>
#include <sstream>
#include <iostream>

namespace quadrants {

// PythonPrintBuffer holds the logs printed from kernel before sending them back
// to python. The name could be a bit misleading, as it is really just a string
// buffer, and can be used without Python. The CPU thread pool executes kernel
// prints concurrently (flex FPS kernels print per-block), so the buffer path
// needs a mutex; the std::cout fallback is thread-safe by the C++ standard.
struct PythonPrintBuffer {
  std::stringstream ss;
  bool enabled{false};
  std::mutex mu;

  template <typename T>
  PythonPrintBuffer &operator<<(const T &t) {
    if (enabled) {
      std::lock_guard<std::mutex> guard(mu);
      ss << t;
    } else {
      std::cout << t;
    }
    return *this;
  }
  std::string pop_content() {
    std::lock_guard<std::mutex> guard(mu);
    auto ret = ss.str();
    ss = std::stringstream();
    return ret;
  }
};

extern PythonPrintBuffer py_cout;

}  // namespace quadrants
