/*
 * Parallel mergesort non recursive with support of threadpool
 * Copyright © 2021 Julián Burella Pérez

 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef PARA_SORT_HPP
#define PARA_SORT_HPP

#ifdef USE_THREAD_POOL
#include <common/ctpl_stl.h>
#endif

#include <algorithm>
#include <future>
#include <thread>
#include <vector>

namespace para_sort
{
template <typename Iter, typename Comp>
void sort(
    Iter begin, Iter end,
    Comp comp,
    unsigned int N = std::thread::hardware_concurrency() / 2
#if defined(USE_THREAD_POOL)
    ,
    ctpl::thread_pool* p = nullptr)
#else
)
#endif
{
    auto len = std::distance(begin, end);
    if (len <= 1024 || N < 2) {
        std::sort(begin, end, comp);
        return;
    }

    const size_t increment = len / N;

#if defined(USE_THREAD_POOL)
    std::vector<std::future<void>> futures;
    futures.reserve(N - 1);
#else
    std::vector<std::thread> threads;
    threads.reserve(N - 1);
#endif

    /* Sorting */
    for (size_t i = 1; i < N; ++i) {
        auto from = begin + i * increment;
        auto to = (i < (N - 1)) ? begin + (i + 1) * increment : end;

#if defined(USE_THREAD_POOL)
        futures.emplace_back(
            p->push([&, from, to](int idx) { std::sort(from, to, comp); }));
#else
        threads.emplace_back([&, from, to]() { std::sort(from, to, comp); });
#endif
    }

    std::sort(begin, begin + increment, comp);
#if defined(USE_THREAD_POOL)
    for (auto& fut : futures)
        fut.get();
#else
    for (auto& th : threads)
        th.join();
#endif

    /* Merging */
    size_t nb_chunks = N;
    size_t chunk_size = increment;
    size_t mid_off = chunk_size;

    while (nb_chunks > 2) {
        const bool is_even = (nb_chunks & 1) == 0;
        mid_off += chunk_size;
        chunk_size *= 2;
        nb_chunks /= 2;
#if defined(USE_THREAD_POOL)
        futures.clear();
#else
        threads.clear();
#endif

        if (nb_chunks > 1) {
            for (size_t j = 1; j < nb_chunks; ++j) {
                bool is_last = j == (nb_chunks - 1);
                auto from = begin + chunk_size * j;
                auto mid = from + chunk_size / 2;
                auto to = (is_last && is_even) ? end : from + chunk_size;

#if defined(USE_THREAD_POOL)
                futures.emplace_back(p->push([&, from, mid, to](int idx) {
                    std::inplace_merge(from, mid, to, comp);
                }));
#else
                threads.emplace_back([&, from, mid, to]() {
                    std::inplace_merge(from, mid, to, comp);
                });
#endif
            }
        }

        std::inplace_merge(begin, begin + chunk_size / 2, begin + chunk_size,
                           comp);

#if defined(USE_THREAD_POOL)
        for (auto& fut : futures)
            fut.get();
#else
        for (auto& th : threads)
            th.join();
#endif
        nb_chunks += is_even ? 0 : 1;
    }

    std::inplace_merge(begin, begin + mid_off, end, comp);
}
}  // namespace para_sort
#endif /* ifndef PARA_SORT_HPP */
