/*
 Concurrent hash map using Junction library as a backend.
 Copyright © 2021 Julián Burella Pérez

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <junction/ConcurrentMap_Leapfrog.h>

namespace concurrent_hash_map
{
template <class K, class D>
class TrivialIterator
{
private:
    std::pair<K, D> pair;

public:
    using key_type = K;
    using mapped_type = D;
    using value_type = std::pair<K, D>;
    using reference = value_type&;

    TrivialIterator(const key_type& k = key_type(),
                    const mapped_type& d = mapped_type())
        : pair(k, d)
    {
    }

    const std::pair<K, D>& operator*() { return pair; }
    std::pair<K, D>* operator->() { return &pair; }

    inline bool operator==(const TrivialIterator& r) const
    {
        return pair.first == r.pair.first;
    }
    inline bool operator!=(const TrivialIterator& r) const
    {
        return pair.first != r.pair.first;
    }
};

template <typename T>
struct ValueTraits {
    using IntType = T;

    /* NOTE
     * A non null value was tried with success but
     * performance where impacted between 3-6 %, there is
     * not a clear explanation for this behavior but we suppose
     * it is related to some optimization with 0 values
     * in the processor
     */
    static constexpr IntType NullValue = 0;
    static constexpr IntType Redirect = -1;
};

template <typename T>
constexpr T ValueTraits<T>::NullValue;

template <typename T>
constexpr T ValueTraits<T>::Redirect;

template <class Key, class T, class H, class E>
class junction_leapfrog_hm
{
   /* NOTE
    * This hash map is an interface to Junction::ConcurrentMap_Leapfrog 
    * implementation of a lock-free hash map. It currently has some 
    * limitations as follow:
    * - Does not support 0 key/value, to fix this, allow values inserted
    *   in the hash map are increased by 1. And when retrieved values
    *   are decreased by one. User are not impacted it is just for
    *   future developers of the library
    * - The maximal number of values supported is 2^64 - 2, we need to
    *   substract NullValue and Redirect value.
    */
private:
    using junc_dflt_type = junction::DefaultKeyTraits<Key>;
    using internal_table_type =
        junction::ConcurrentMap_Leapfrog<Key, T, junc_dflt_type,
                                         ValueTraits<T>>;
    std::unique_ptr<internal_table_type> hash;
    junction::QSBR::Context qsbrContext;
    std::mutex g_i_mutex;

    size_t next_power_2(size_t n)
    {
        size_t temp = 1;
        while (n >= temp)
            temp <<= 1;
        return temp;
    }

public:
    using key_type = Key;
    using mapped_type = T;
    using value_type = std::pair<const key_type, mapped_type>;
    using iterator = TrivialIterator<key_type, mapped_type>;
    using insert_return_type = std::pair<iterator, bool>;

    const T value(iterator it) const { return it->second; }
    junction_leapfrog_hm() : hash(std::make_unique<internal_table_type>())
    {
        qsbrContext = junction::DefaultQSBR.createContext();
    }
    junction_leapfrog_hm(size_t cap)
        : hash(std::make_unique<internal_table_type>(next_power_2(cap) << 1))
    {
        qsbrContext = junction::DefaultQSBR.createContext();
    }
    ~junction_leapfrog_hm()
    {
        junction::DefaultQSBR.destroyContext(qsbrContext);
    }

    junction_leapfrog_hm(const junction_leapfrog_hm&) = delete;
    junction_leapfrog_hm& operator=(const junction_leapfrog_hm&) = delete;

    junction_leapfrog_hm(junction_leapfrog_hm&& rhs) = default;
    junction_leapfrog_hm& operator=(junction_leapfrog_hm&& other)
    {
        if (this != &other) {
            this->hash = std::move(other.hash);
        }
        return *this;
    }

    iterator find(const key_type& k)
    {
        std::lock_guard<std::mutex> lock(g_i_mutex);
        mapped_type r = hash->get(k + 1);

        if (r != ValueTraits<T>::NullValue)
            return iterator(k, r - 1);
        else
            return end();
    }

    insert_return_type insert(const key_type& k, const mapped_type& d)
    {
        std::lock_guard<std::mutex> lock(g_i_mutex);
        auto mutator = hash->insertOrFind(k + 1);
        auto inserted = mutator.getValue() == ValueTraits<T>::NullValue;

        if (inserted)
            mutator.assignValue(d + 1);

        return insert_return_type(iterator(k, d), inserted);
    }

    insert_return_type insert(const value_type& d)
    {
        return insert(d.first, d.second);
    }

    bool update(iterator& it, T& expected, T desired)
    {
        std::lock_guard<std::mutex> lock(g_i_mutex);
        return hash->exchange(it->first + 1, desired + 1) == (expected + 1);
    }

    void quiescent(void) { junction::DefaultQSBR.update(qsbrContext); }
    iterator end() { return iterator(-1); }
    void reserve(size_t hint) {}
    template <class F>
    void foreach (const F& f) const
    {
        for (typename internal_table_type::Iterator it(*hash); it.isValid();
             it.next()) {
            f(value_type(it.getKey() - 1, it.getValue() - 1));
        }
    }
};

}  // namespace concurrent_hash_map
