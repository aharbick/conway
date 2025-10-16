#ifndef _GOL_MEMORY_H_
#define _GOL_MEMORY_H_

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "cuda_utils.h"
#include "subgrid_cache.h"

namespace gol {

// CUDA Device Memory RAII Wrapper
class CudaMemory {
 private:
  void* ptr_;
  size_t size_;

 public:
  CudaMemory(size_t size) : ptr_(nullptr), size_(size) {
    cudaCheckError(cudaMalloc(&ptr_, size));
  }

  ~CudaMemory() {
    if (ptr_) {
      cudaFree(ptr_);  // cudaFree handles nullptr safely
    }
  }

  // Move semantics only (no copying CUDA memory)
  CudaMemory(CudaMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  CudaMemory& operator=(CudaMemory&& other) noexcept {
    if (this != &other) {
      if (ptr_)
        cudaFree(ptr_);
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Delete copy constructor and assignment
  CudaMemory(const CudaMemory&) = delete;
  CudaMemory& operator=(const CudaMemory&) = delete;

  void* get() const {
    return ptr_;
  }
  size_t size() const {
    return size_;
  }

  template <typename T>
  T* as() const {
    return static_cast<T*>(ptr_);
  }
};

// Host Memory RAII using smart pointers with custom deleters
template <typename T>
using HostPtr = std::unique_ptr<T, void (*)(void*)>;

template <typename T>
HostPtr<T> make_host_ptr(size_t count = 1) {
  T* ptr = static_cast<T*>(malloc(sizeof(T) * count));
  if (!ptr) {
    throw std::bad_alloc();
  }
  return HostPtr<T>(ptr, free);
}

template <typename T>
HostPtr<T> make_host_ptr_zeroed(size_t count = 1) {
  T* ptr = static_cast<T*>(calloc(count, sizeof(T)));
  if (!ptr) {
    throw std::bad_alloc();
  }
  return HostPtr<T>(ptr, free);
}

// SearchMemory with RAII
class SearchMemory {
 private:
  // CUDA device memory
  CudaMemory d_candidates_;
  CudaMemory d_numCandidates_;
  CudaMemory d_bestPattern_;
  CudaMemory d_bestGenerations_;

  // Host memory
  HostPtr<uint64_t> h_candidates_;
  HostPtr<uint64_t> h_numCandidates_;
  HostPtr<uint64_t> h_bestPattern_;
  HostPtr<uint64_t> h_bestGenerations_;

  // Cache hash table (borrowed reference, owned by SubgridCache singleton)
  SubgridHashTable* d_cacheTable_;

 public:
  SearchMemory(size_t candidateSize)
      : d_candidates_(sizeof(uint64_t) * candidateSize),
        d_numCandidates_(sizeof(uint64_t)),
        d_bestPattern_(sizeof(uint64_t)),
        d_bestGenerations_(sizeof(uint64_t)),
        h_candidates_(make_host_ptr_zeroed<uint64_t>(candidateSize)),
        h_numCandidates_(make_host_ptr<uint64_t>()),
        h_bestPattern_(make_host_ptr<uint64_t>()),
        h_bestGenerations_(make_host_ptr<uint64_t>()),
        d_cacheTable_(nullptr) {
    // Constructor automatically handles all allocations
    // Note: d_cacheTable_ is a borrowed reference and not freed here
  }

  // Move semantics only (expensive to copy large memory buffers)
  SearchMemory(SearchMemory&&) = default;
  SearchMemory& operator=(SearchMemory&&) = default;

  // Delete copy semantics
  SearchMemory(const SearchMemory&) = delete;
  SearchMemory& operator=(const SearchMemory&) = delete;

  // Accessors for compatibility with existing code
  uint64_t* d_candidates() const {
    return d_candidates_.as<uint64_t>();
  }
  uint64_t* d_numCandidates() const {
    return d_numCandidates_.as<uint64_t>();
  }
  uint64_t* d_bestPattern() const {
    return d_bestPattern_.as<uint64_t>();
  }
  uint64_t* d_bestGenerations() const {
    return d_bestGenerations_.as<uint64_t>();
  }

  uint64_t* h_candidates() const {
    return h_candidates_.get();
  }
  uint64_t* h_numCandidates() const {
    return h_numCandidates_.get();
  }
  uint64_t* h_bestPattern() const {
    return h_bestPattern_.get();
  }
  uint64_t* h_bestGenerations() const {
    return h_bestGenerations_.get();
  }

  // Cache table accessor with lazy loading
  // Automatically builds GPU hash table from SubgridCache if loaded
  // Returns borrowed reference (owned by SubgridCache singleton)
  SubgridHashTable* d_cacheTable() {
    if (d_cacheTable_ == nullptr && SubgridCache::getInstance().isLoaded()) {
      d_cacheTable_ = SubgridCache::getInstance().getGpuHashTable();
    }
    return d_cacheTable_;
  }
};

}  // namespace gol

#endif