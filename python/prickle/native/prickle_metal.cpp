#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "prickle_metal.h"

static MTL::Device *device;
static MTL::CommandQueue *cq;
static MTL::CommandBuffer *cb;
static MTL::ComputeCommandEncoder *ce;
static int64_t queue_cap = 0;
static int64_t queue_size = 0;

extern "C" void prickle_init(int64_t queue_capacity) {
  if (device == nullptr) {
    device = MTL::CreateSystemDefaultDevice();
    queue_cap = queue_capacity;
    cq = device->newCommandQueue(queue_cap);
    if (cq == nullptr) {
      fprintf(stderr, "Failed to set up command queue\n");
      exit(1);
    }
  }
}

extern "C" void prickle_sync() {
  prickle_metal::sync();
}

extern "C" MTL::Buffer *prickle_alloc_buffer(int64_t nbytes) {
  MTL::Buffer *buf = device->newBuffer(nbytes, MTL::ResourceStorageModeShared);
  if (buf == nullptr) {
    fprintf(stderr, "Failed to allocate buffer of %lld bytes\n", nbytes);
    exit(1);
  }
  return buf;
}

extern "C" void *prickle_ptr_buffer(MTL::Buffer *buf) {
  return buf->contents();
}

extern "C" void prickle_memcpy(void *dst, void *src, int64_t nbytes) {
  memcpy(dst, src, nbytes);
}

extern "C" void prickle_free_buffer(MTL::Buffer *buf) {
  buf->release();
}

namespace prickle_metal {
  MTL::Library *load_library(const char *lib_str) {
    NS::String *code = NS::String::string(lib_str, NS::ASCIIStringEncoding);
    NS::Error *err;
    MTL::Library *lib = device->newLibrary(code, nullptr, &err);
    if (lib == nullptr) {
      fprintf(stderr, "Error compiling library: %s\n", err->description()->utf8String());
      exit(1);
    }
    return lib;
  }

  MTL::Function *get_fun(MTL::Library *lib, const char *fun_id) {
    NS::String *s = NS::String::string(fun_id, NS::ASCIIStringEncoding);
    MTL::Function *f = lib->newFunction(s);
    if (f == nullptr) {
      fprintf(stderr, "Could not find function %s in library\n", fun_id);
      exit(1);
    }
    return f;
  }

  MTL::Buffer *alloc(int64_t nbytes) {
    return prickle_alloc_buffer(nbytes);
  }

  void free(MTL::Buffer *b) {
    prickle_free_buffer(b);
  }

  void copy(void *dst, void *src, int64_t nbytes, int64_t k) {
    // If an argument represents device memory, it is an MTL::Buffer pointer
    // from which we need to extract the actual data pointer. Otherwise, we use
    // the provided pointer immediately. We use 'k' to encode the memory types
    // of the arguments:
    //  0: both host
    //  1: source is in host memory, destination on device
    //  2: source is in device memory, destination on host
    //  3: both device
    dst = k & 1 ? ((MTL::Buffer*)dst)->contents() : dst;
    src = k & 2 ? ((MTL::Buffer*)src)->contents() : src;
    prickle_memcpy(dst, src, nbytes);
  }

  void launch_kernel(
      MTL::Function *kernel,
      std::vector<MTL::Buffer*> args,
      int64_t block_x, int64_t block_y, int64_t block_z,
      int64_t thread_x, int64_t thread_y, int64_t thread_z) {
    if (cb == nullptr || cb->status() != MTL::CommandBufferStatusNotEnqueued) {
      if (cb != nullptr) cb->release();
      cb = cq->commandBuffer();
      if (cb == nullptr) {
        fprintf(stderr, "Failed to set up command buffer\n");
        exit(1);
      }
    }

    if (ce == nullptr) {
      ce = cb->computeCommandEncoder();
      if (ce == nullptr) {
        fprintf(stderr, "Failed to set up compute command encoder\n");
        exit(1);
      }
    }

    NS::Error *err;
    MTL::ComputePipelineState *state = device->newComputePipelineState(kernel, &err);
    if (state == nullptr) {
      fprintf(stderr, "Error setting up compute pipeline state: %s\n", err->description()->utf8String());
      exit(1);
    }

    ce->setComputePipelineState(state);
    for (int i = 0; i < args.size(); i++) {
      ce->setBuffer(args[i], 0, i);
    }

    int simd_width = state->threadExecutionWidth();
    if (simd_width != 32) {
      fprintf(stderr, "Expected SIMD width of 32, found %d which is not supported\n", simd_width);
      exit(1);
    }
    NS::UInteger maxthreads = state->maxTotalThreadsPerThreadgroup();
    assert(thread_x * thread_y * thread_z <= maxthreads);

    MTL::Size grid_size = MTL::Size::Make(block_x * thread_x, block_y * thread_y, block_z * thread_z);
    MTL::Size block_size = MTL::Size::Make(thread_x, thread_y, thread_z);
    ce->dispatchThreads(grid_size, block_size);
    if (++queue_size == queue_cap) {
      submit_work();
    }
  }

  void submit_work() {
    if (ce != nullptr) {
      ce->endEncoding();
      cb->commit();
      cb->waitUntilScheduled();
      ce->release();
      ce = nullptr;
      queue_size = 0;
    }
  }

  void sync() {
    submit_work();
    if (cb != nullptr) {
      cb->waitUntilCompleted();
      cb->release();
      cb = nullptr;
    }
  }
}
