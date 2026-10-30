#include "quadrants/runtime/program_impls/metal/metal_program.h"

#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/codegen/spirv/kernel_compiler.h"
#include "quadrants/codegen/spirv/compiled_kernel_data.h"
#include "quadrants/rhi/metal/metal_device.h"
#include "quadrants/runtime/gfx/snode_tree_manager.h"
#include "quadrants/runtime/gfx/kernel_launcher.h"
#include "quadrants/rhi/common/host_memory_pool.h"

namespace quadrants::lang {

MetalProgramImpl::MetalProgramImpl(CompileConfig &config) : GfxProgramImpl(config) {
}

void MetalProgramImpl::materialize_runtime(KernelProfilerBase *profiler, uint64 **result_buffer_ptr) {
  *result_buffer_ptr =
      (uint64 *)HostMemoryPool::get_instance().allocate(sizeof(uint64) * quadrants_result_buffer_entries, 8);

  metal::MetalDevice *mtl_dev = nullptr;
  if (config->external_metal_command_queue != 0) {
    mtl_dev = metal::MetalDevice::create_with_external_queue(config->external_metal_command_queue);
  } else {
    mtl_dev = metal::MetalDevice::create();
  }
  // A69: forward the Program-level fast_math flag to the MSL compiler (device-level
  // setting; one qd.init == one Program == one device). Default true keeps old behaviour.
  mtl_dev->set_fast_math(config->fast_math);
  device_ = std::unique_ptr<metal::MetalDevice>(mtl_dev);

  gfx::GfxRuntime::Params params;
  params.device = device_.get();
  params.program_impl = this;
  // A72: wire the Program-level kernel profiler into GfxRuntime (Vulkan already
  // did); attribution then flows through per-task scopes -> per-buffer
  // GPUStartTime/GPUEndTime -> profiler_flush_sampled_time -> insert_record.
  params.profiler = profiler;
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void MetalProgramImpl::enqueue_compute_op_lambda(std::function<void(Device *device, CommandList *cmdlist)> op,
                                                 const std::vector<ComputeOpImageRef> &image_refs) {
  runtime_->enqueue_compute_op_lambda(op, image_refs);
}

}  // namespace quadrants::lang
