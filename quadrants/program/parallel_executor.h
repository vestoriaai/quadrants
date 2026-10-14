#pragma once

#include <exception>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include "quadrants/common/core.h"

namespace quadrants::lang {
class ParallelExecutor {
 public:
  using TaskType = std::function<void()>;

  explicit ParallelExecutor(const std::string &name, int num_threads);
  ~ParallelExecutor();

  void enqueue(const TaskType &func);

  void flush();


  int get_num_threads() {
    return num_threads_;
  }

 private:
  enum class ExecutorStatus {
    uninitialized,
    initialized,
    finalized,
  };

  void worker_loop();

  // Must be called while holding |mut|.
  bool flush_cv_cond();

  std::string name_;
  int num_threads_;
  std::atomic<int> thread_counter_{0};
  std::mutex mut_;

  // All guarded by |mut|
  ExecutorStatus status_;
  std::vector<std::thread> threads_;
  std::deque<TaskType> task_queue_;
  // A21: message of the first error thrown by a task inside a worker thread.
  // Tasks must not let exceptions escape (worker threads have no unwinding
  // contract; an escaping throw hits std::terminate -> SIGABRT). The message
  // is stashed here and re-raised as std::runtime_error by flush() in the
  // submitting thread, where it crosses the python boundary normally.
  std::string pending_error_;

  int running_threads_;

  // Used to signal the workers that they can start polling from |task_queue|.
  std::condition_variable init_cv_;
  // Used by |this| to instruct the worker thread that there is an event:
  // * task being enqueued
  // * shutting down
  std::condition_variable worker_cv_;
  // Used by a worker thread to unblock the caller from waiting for a flush.
  //
  // TODO: Instead of having this as a member variable, we can enqueue a
  // callback upon flush(). The flush() will then block waiting for that
  // callback to be executed?
  std::condition_variable flush_cv_;
};
}  // namespace quadrants::lang
