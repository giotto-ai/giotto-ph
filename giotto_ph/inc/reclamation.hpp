#pragma once

#include <atomic>
#include <iostream>
#include <vector>
#include <mutex>

namespace mrzv
{
template <class T>
struct MemoryManager {
    std::vector<T*> retiring_;
    std::atomic<int> counter_;
    size_t n_threads_;
    bool even_epoch_ = false;
    std::mutex mut;

    MemoryManager(unsigned n_threads)
        : counter_(0), n_threads_(n_threads), retiring_(n_threads)
    {
    }

    ~MemoryManager()
    {
        for (T* p : retiring_) {
            delete p;
        }
    }
    bool is_even_epoch(int counter) const
    {
        return (counter / n_threads_) % 2 == 0;
    }
    void retire(T* ptr)
    {
        if (ptr) {
            std::lock_guard<std::mutex> lock(mut);
            retiring_.push_back(ptr);
        }
    }
    void quiescent()
    {
        if (even_epoch_ != is_even_epoch(counter_)) {
            ++counter_;
            even_epoch_ = !even_epoch_;
            std::vector<T*> to_delete;
            {
                std::lock_guard<std::mutex> lock(mut);
                retiring_.swap(to_delete);
            }
            for (T* p : to_delete) {
                delete p;
            }
        }
    }
};

}  // namespace mrzv
