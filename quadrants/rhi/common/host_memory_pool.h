#pragma once
#include "quadrants/common/core.h"
#include "quadrants/rhi/common/unified_allocator.h"
#include "quadrants/rhi/device.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <memory>
#include <thread>

namespace quadrants::lang {

// A memory pool that runs on the host

class QD_DLL_EXPORT HostMemoryPool {
 public:
  static const size_t page_size;

  static HostMemoryPool &get_instance();

  void *allocate(std::size_t size, std::size_t alignment, bool exclusive = false);
  void release(std::size_t size, void *ptr);
  void reset();
  HostMemoryPool();
  ~HostMemoryPool();

  // A60: bump-allocator generation. `reset()` (invoked from Program::finalize on
  // every qd.reset()/re-init) unmaps EVERY chunk the pool owns. A new generation
  // then re-mmaps and the OS frequently recycles the just-freed addresses, so a
  // late `~Ndarray` from the old generation can resolve its stale pointer inside
  // the NEW allocator and munmap the new generation's live page (ABA -> SIGBUS).
  // Devices stamp each allocation with the generation it was made under and skip
  // releases whose stamp no longer matches; see CpuDevice::dealloc_memory.
  uint64_t generation() const { return generation_.load(std::memory_order_relaxed); }

 protected:
  void *allocate_raw_memory(std::size_t size);
  void deallocate_raw_memory(void *ptr);

  // All the raw memory allocated from OS/Driver
  // We need to keep track of them to guarantee that they are freed
  std::map<void *, std::size_t> raw_memory_chunks_;

  std::unique_ptr<UnifiedAllocator> allocator_;
  std::mutex mut_allocation_;
  std::atomic<uint64_t> generation_{0};

  friend class UnifiedAllocator;
};

}  // namespace quadrants::lang
