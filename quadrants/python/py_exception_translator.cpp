#include <csignal>
#include <exception>
#include <cstdio>
#include <string>

#include <nanobind/nanobind.h>

#include "quadrants/python/export.h"

namespace quadrants {

// Translates and propagates a CPP exception to Python.
//
// Unlike pybind11, nanobind does not support calling its API (including register_exception_translator) from a
// global/static constructor: doing so runs before the extension's NB_MODULE init has set up nanobind's internals
// and crashes during dlopen. So this is now an explicit function invoked from NB_MODULE (see export.cpp).
void register_py_exception_translator() {
  // A21 diagnostics: uncaught C++ exceptions terminate with the message lost
  // (buffered). Surface it on the unbuffered stderr before dying.
  std::set_terminate([]() {
    // A21: write to a FILE (explicitly closed) - every pipe/buffer variant has
    // proven lossy when abort() fires during compilation.
    FILE *diag = fopen("/tmp/quadrants_terminate.log", "w");
    if (auto p = std::current_exception()) {
      try {
        std::rethrow_exception(p);
      } catch (const std::string &e) {
        if (diag) fprintf(diag, "QUADRANTS_TERMINATE std::string: %s\n", e.c_str());
      } catch (const std::exception &e) {
        if (diag) fprintf(diag, "QUADRANTS_TERMINATE exception: %s\n", e.what());
      } catch (...) {
        if (diag) fprintf(diag, "QUADRANTS_TERMINATE unknown exception\n");
      }
    } else {
      if (diag) fprintf(diag, "QUADRANTS_TERMINATE no active exception\n");
    }
    if (diag) fclose(diag);
    std::abort();
  });
  nanobind::register_exception_translator([](const std::exception_ptr &p, void * /*payload*/) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const std::string &e) {
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
    }
  });
}

}  // namespace quadrants
