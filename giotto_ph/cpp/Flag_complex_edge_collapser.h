/* This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which
 * is released under MIT. See file LICENSE or go to
 * https://gudhi.inria.fr/licensing/ for full license details. Author(s):
 * Siddharth Pritam
 *
 *    Copyright (C) 2020 Inria
 *
 *    Modification(s):
 *      - 2020/03 Vincent Rouvreau: integration to the gudhi library
 *      - 2021/01 Julián Burella Pérez:
 *          * Remove dependencies with EIGEN and Boost
 *          * Remove `Neighbours` make iterators explicit
 *          * Remove use of `unordered_map`
 *          * Replace `std::set_intersection` by custom method, it's not faster,
                but it allow's to process directly each vertex found
 *          * Remove `critical_edge_indicator_`, now this information is stored 
 *              directly into the Sparse_row_matrix second element (index)
 *              the MSB correspond to a flag if the index was already indicated
 *              as a critical edge
 */
#ifndef FLAG_COMPLEX_EDGE_COLLAPSER_H_
#define FLAG_COMPLEX_EDGE_COLLAPSER_H_

#include <algorithm>  // for std::includes
#include <iostream>
#include <iterator>  // for std::inserter
#include <numeric>
#include <set>
#include <tuple>        // for std::tie
#include <type_traits>  // for std::decay
#include <vector>

namespace Gudhi
{
namespace collapse
{
/** \private
 *
 * \brief Flag complex sparse matrix data structure.
 *
 * \details
 * TODO
 *
 * \tparam Vertex type must be a signed integer type. It admits a total order <.
 * \tparam Filtration type for the value of the filtration function. Must be
 * comparable with <.
 */
template <typename vertex_t, typename filt_t>
class Flag_complex_edge_collapser
{
private:
    // internal numbering of edges
    using edge_idx_t = vertex_t;

    // The sparse matrix data type
    // My profiling showed that using a `std::pair` was a bit slower in this
    // case
    using edge_t = struct pair_ {
        vertex_t first;
        edge_idx_t second;
    };
    using Sparse_vector = std::vector<edge_t>;
    using Sparse_row_matrix = std::vector<Sparse_vector>;

    // Range of neighbors of a vertex a range of row indices
    using vertex_t_vector = std::vector<vertex_t>;

    const vertex_t max_val = std::numeric_limits<vertex_t>::max();
    static constexpr edge_idx_t edge_max_val =
        std::numeric_limits<edge_idx_t>::max();

public:
    /** \brief filt_edge_t is a type to store an edge with its filtration
     * value. */
    using filt_edge_t = std::tuple<vertex_t, vertex_t, filt_t>;

private:
    /* index translation for unordered vertices in the input data */
    vertex_t row_to_vertex_ = 0;
    std::vector<uint8_t> dominated_edge_idx;

    std::vector<vertex_t> vertex_to_row_;

    // Stores the Sparse matrix of Filtration values representing the original
    // graph. The matrix rows and columns are indexed by vertex_t.
    Sparse_row_matrix sparse_row_adjacency_matrix_;

    // The input, a vector of filtered edges.
    std::vector<filt_edge_t> f_edge_vector_;

    const edge_idx_t nb_edges;
    vertex_t nb_vertex;

    static constexpr edge_idx_t msb_bit = (1UL << (sizeof(edge_idx_t) * 8 - 1));
    static constexpr edge_idx_t msb_mask = ~msb_bit;

    enum filt_comp_t { u = 0, v = 1 };

    /* Returns vertex of the corresponding edge of input edge */
    inline const vertex_t edge_vertex(const edge_idx_t idx_edge,
                                      const filt_comp_t elem) noexcept
    {
        if (elem == filt_comp_t::u)
            return std::get<filt_comp_t::u>(f_edge_vector_[idx_edge]);
        return std::get<filt_comp_t::v>(f_edge_vector_[idx_edge]);
    }

    /* Returns filtration of the corresponding edge of input edge */
    inline const filt_t edge_filt(const edge_idx_t idx_edge) noexcept
    {
        return std::get<2>(f_edge_vector_[idx_edge]);
    }

    /* This method returns the correponding column index of the
     * sparse_row_adjancency_matrix_ for the vertex of the original input */
    inline const vertex_t adj_mat_col_idx(const vertex_t vertex_idx) noexcept
    {
        return vertex_to_row_[vertex_idx];
    }

    /* Checks if the edge has been marked as a critical edge */
    inline const edge_idx_t
    mask_critical_edge(const edge_idx_t& val) const noexcept
    {
        return val & msb_mask;
    }

#define THRESH_SEARCH 16
    inline auto find_edge(const edge_idx_t u, const edge_idx_t v) noexcept
    {
        if (sparse_row_adjacency_matrix_[u].size() > THRESH_SEARCH)
            return std::lower_bound(
                sparse_row_adjacency_matrix_[u].begin(),
                sparse_row_adjacency_matrix_[u].end(), v,
                [&](const auto& a, const auto& b) { return a.first < b; });

        auto it = sparse_row_adjacency_matrix_[u].begin();
        for (; it != sparse_row_adjacency_matrix_[u].end() && it->first < v;
             ++it)
            ;

        return it;
    }

    inline void set_critical_edge_index(const edge_idx_t u,
                                        const edge_idx_t v) noexcept
    {
        find_edge(u, v)->second |= msb_bit;
    }

    inline const bool critical_edge_index(const edge_idx_t& idx) const noexcept
    {
        return idx & msb_bit;
    }

    /* test if the vertex is valid in the Neighbourhood open condition*/
    inline const bool valid_it_open(const edge_idx_t& idx,
                                    const edge_idx_t& current_backward) noexcept
    {
        return mask_critical_edge(idx) < current_backward ||
               critical_edge_index(idx);
    }

    /* left and right iterators are not exactly the same */
    /* The right iterator depends on the current value of the left
     * iterator */
    /* these iterators are used for the intersection */
    inline typename Sparse_vector::iterator&
    next_left(typename Sparse_vector::iterator& start,
              const typename Sparse_vector::const_iterator& end,
              const edge_idx_t& current_backward) noexcept
    {
        for (; start != end; ++start)
            if (valid_it_open(start->second, current_backward))
                return start;
        return start;
    };

    /* these iterators are used for the intersection */
    inline typename Sparse_vector::iterator&
    next_right(typename Sparse_vector::iterator& start,
               const typename Sparse_vector::const_iterator& end,
               const vertex_t& left_vertex,
               const edge_idx_t& current_backward) noexcept
    {
        for (; start != end; ++start)
            if (start->first >= left_vertex &&
                valid_it_open(start->second, current_backward))
                return start;
        return start;
    };

    template <typename Func>
    void custom_intersection(const vertex_t rw_u, const vertex_t rw_v,
                             const edge_idx_t curr,
                             const Func& callback) noexcept
    {
        /* prepare iterators for the intersection */
        auto& left_ref = sparse_row_adjacency_matrix_[rw_u];
        const auto& left_end = left_ref.cend();
        auto& right_ref = sparse_row_adjacency_matrix_[rw_v];
        const auto& right_end = right_ref.cend();

        /* Initialize iterators */
        auto start_left = left_ref.begin();
        auto start_right = right_ref.begin();
        auto left = next_left(start_left, left_end, curr);
        auto right = next_right(start_right, right_end, left->first, curr);

        bool inc_left_it;

        while (left != left_end && right != right_end) {
            inc_left_it = false;
            if (left->first == right->first) {
                /* union */
                const auto vertex = left->first;
                inc_left_it = true;

                callback(vertex);
            } else if (left->first < right->first) {
                inc_left_it = true;
            } else if (left != left_end) {
                ++right;
                right = next_right(right, right_end, left->first, curr);
            }

            if (inc_left_it) {
                ++left;
                left = next_left(left, left_end, curr);
                // left = next_right(left, left_end, right->first, curr);
            }
        }
    }

    const bool edge_is_dominated(const vertex_t rw_u, const vertex_t rw_v,
                                 const edge_idx_t curr) noexcept
    {
        /* test if the vertex is valid in the Neighbourhood close condition*/
        auto valid_it_closed = [&](const auto& idx) noexcept {
            return valid_it_open(idx, curr) ||
                   mask_critical_edge(idx) == nb_edges;
        };

        /* iterator for the union operator */
        /* the left, right iterators are not the real bottleneck */
        /* the iterator for the union is the bottleneck because when it computes
         * for a different vertex, it needs to computes with all the previous
         * vertex generated by the intersection, and this each time the
         * vertex does change
         */
        auto next_it = [&](auto& start, const auto& end, const auto& vertex,
                           const auto& valid) noexcept ->
            typename Sparse_vector::iterator& {
                for (; start != end; ++start) {
                    if (start->first >= vertex) {
                        if (start->first != vertex || !valid(start->second)) {
                            start = end;
                        }
                        break;
                    }
                }

                return start;
            };

        vertex_t_vector intersection_vec;
        typename Sparse_vector::iterator neig_it;
        typename Sparse_vector::iterator end_it = neig_it;
        vertex_t idx_ongoing = 0;

        /* Allocated the maximal possible number of vertices in the
         * intersection */
        intersection_vec.reserve(
            std::min(sparse_row_adjacency_matrix_[rw_u].size(),
                     sparse_row_adjacency_matrix_[rw_v].size()));

        custom_intersection(rw_u, rw_v, curr, [&](auto& vertex) {
            intersection_vec.push_back(vertex);
            /* If the iterator is valid, check if the new vertex is
             * contained inside of the iterator remaining elements
             */
            if (neig_it != end_it) {
                neig_it = next_it(neig_it, end_it, vertex, valid_it_closed);
            }

            while ((neig_it == end_it) &&
                   idx_ongoing < intersection_vec.size()) {
                /* Previous iterator was invalid, select next one
                 * available
                 */
                neig_it =
                    sparse_row_adjacency_matrix_[intersection_vec[idx_ongoing]]
                        .begin();
                end_it =
                    sparse_row_adjacency_matrix_[intersection_vec[idx_ongoing]]
                        .end();

                /* For the new neighbour iterator, check all previous
                 * vertex extracted
                 */
                for (auto& elem : intersection_vec) {
                    neig_it = next_it(neig_it, end_it, elem, valid_it_closed);

                    if (neig_it == end_it) {
                        idx_ongoing++;
                        break;
                    }
                }
            }
        });

        /* If any Neighbours is alive, then return true */
        return (intersection_vec.size() == 1) || (neig_it != end_it);
    }

    // insert to edges_idx the edges connecting u and v (extremities of crit) to
    // their common neighbors (not themselves)
    inline void three_clique_indices(
        std::set<edge_idx_t, std::greater<edge_idx_t>>& edge_idx,
        const edge_idx_t thresh, const vertex_t rw_u,
        const vertex_t rw_v) noexcept
    {
        custom_intersection(rw_u, rw_v, thresh, [&](auto& vertex) {
            auto&& e_v = std::minmax(rw_u, vertex);
            auto&& e_u = std::minmax(rw_v, vertex);
            auto&& a = find_edge(e_v.first, e_v.second);
            auto&& b = find_edge(e_u.first, e_u.second);

            if (!critical_edge_index(a->second) && a->second < thresh)
                edge_idx.emplace(a->second);
            if (!critical_edge_index(b->second) && b->second < thresh)
                edge_idx.emplace(b->second);
        });
    }

    // Detect and set all edges that are becoming critical
    template <typename FilteredEdgeOutput>
    void set_edge_critical(const edge_idx_t indx, const filt_t filt,
                           const FilteredEdgeOutput filtered_edge_output)
    {
        edge_idx_t current;
        /* the set will contains the indices from greater to lower, this
         * allows to iterate in normal
         */
        std::set<edge_idx_t, std::greater<edge_idx_t>> effected_idx;

        three_clique_indices(
            effected_idx, indx,
            adj_mat_col_idx(edge_vertex(indx, filt_comp_t::u)),
            adj_mat_col_idx(edge_vertex(indx, filt_comp_t::v)));

        for (auto it = effected_idx.begin(); it != effected_idx.end(); ++it) {
            current = mask_critical_edge(*it);
            const auto rw_u =
                adj_mat_col_idx(edge_vertex(current, filt_comp_t::u));
            const auto rw_v =
                adj_mat_col_idx(edge_vertex(current, filt_comp_t::v));
            // If current is not critical so it should be
            // processed, otherwise it stays in the graph
            if (!dominated_edge_idx[current] &&
                !edge_is_dominated(rw_u, rw_v, current)) {
                dominated_edge_idx[current] = true;
                set_critical_edge_index(rw_v, rw_u);
                set_critical_edge_index(rw_u, rw_v);

                filtered_edge_output(edge_vertex(current, filt_comp_t::u),
                                     edge_vertex(current, filt_comp_t::v),
                                     filt);
                three_clique_indices(effected_idx, current, rw_u, rw_v);
            }
        }
    }

    // Insert a vertex in the data structure
    const vertex_t insert_vertex(const vertex_t vertex)
    {
        if (vertex_to_row_[vertex] == max_val) {
            // If it was not already inserted - Value won't be updated by
            // emplace if it is already present
            vertex_to_row_[vertex] = row_to_vertex_;
            // Expand the matrix. The size of rows is irrelevant.
            // Initializing the diagonal element of the adjency matrix
            // corresponding to rw_v.
            sparse_row_adjacency_matrix_.emplace_back(
                Sparse_vector{{row_to_vertex_++, nb_edges}});
            sparse_row_adjacency_matrix_.back().reserve(nb_vertex);
        }

        return adj_mat_col_idx(vertex);
    }
    // Insert an edge in the data structure
    // exception std::invalid_argument In debug mode, if u == v
    void insert_new_edge(const vertex_t& u, const vertex_t& v,
                         const edge_idx_t& idx)
    {
        // The edge must not be added before, it should be a new edge.
        vertex_t rw_u = insert_vertex(u);
        vertex_t rw_v = insert_vertex(v);
        sparse_row_adjacency_matrix_[rw_u].insert(find_edge(rw_u, rw_v),
                                                  {rw_v, idx});
        sparse_row_adjacency_matrix_[rw_v].insert(find_edge(rw_v, rw_u),
                                                  {rw_u, idx});
    }

public:
    /** \brief Flag_complex_edge_collapser constructor from a range of filtered
     * edges.
     *
     * @param[in] edges Range of Filtered edges range.There is no need the range
     * to be sorted, as it will be performed in
     * `Flag_complex_edge_collapser::process_edges`.
     *
     * \tparam FilteredEdgeRange must be a range for which std::begin and
     * std::end return iterators on a `Flag_complex_edge_collapser::filt_edge_t`
     */
    template <typename FilteredEdgeRange>
    Flag_complex_edge_collapser(const FilteredEdgeRange& edges)
        : f_edge_vector_(std::begin(edges), std::end(edges)),
          nb_edges(edges.size())
    {
    }
    /** \brief Performs edge collapse in a increasing sequence of
     * the filtration value.
     *
     * \tparam filtered_edge_output is a functor that is called on the output 
     * edges, in non-decreasing order of filtration, as 
     * filtered_edge_output(u, v, f) where u and v are Vertex representing 
     * the extremities of the edge, and f is its new Filtration.
     */
    template <typename FilteredEdgeOutput>
    void process_edges(FilteredEdgeOutput filtered_edge_output)
    {
        // Sort edges
        auto sort_by_filtration = [](const filt_edge_t& edge_a,
                                     const filt_edge_t& edge_b) -> bool {
            return (std::get<2>(edge_a) < std::get<2>(edge_b));
        };

        auto sort_by_vertex0 = [](const filt_edge_t& edge_a,
                                  const filt_edge_t& edge_b) -> bool {
            return (std::get<0>(edge_a) < std::get<0>(edge_b));
        };
        auto sort_by_vertex1 = [](const filt_edge_t& edge_a,
                                  const filt_edge_t& edge_b) -> bool {
            return (std::get<1>(edge_a) < std::get<1>(edge_b));
        };

        const vertex_t nb_vertex_col0 = std::get<0>(*std::max_element(
            f_edge_vector_.begin(), f_edge_vector_.end(), sort_by_vertex0));
        const vertex_t nb_vertex_col1 = std::get<1>(*std::max_element(
            f_edge_vector_.begin(), f_edge_vector_.end(), sort_by_vertex1));

        nb_vertex = std::max(nb_vertex_col0, nb_vertex_col1) + 1;
        sparse_row_adjacency_matrix_.reserve(nb_vertex);

        vertex_to_row_.resize(nb_vertex, max_val);

        std::sort(f_edge_vector_.begin(), f_edge_vector_.end(),
                  sort_by_filtration);

        dominated_edge_idx.resize(nb_edges);
        std::fill(dominated_edge_idx.begin(), dominated_edge_idx.end(), false);

        /* First edge is processed differently */
        /* The first edge is always not dominated */
        vertex_t u = edge_vertex(0, filt_comp_t::u);
        vertex_t v = edge_vertex(0, filt_comp_t::v);

        insert_new_edge(u, v, 0);
        auto rw_u = adj_mat_col_idx(u);
        auto rw_v = adj_mat_col_idx(v);

        dominated_edge_idx[0] = true;
        set_critical_edge_index(rw_v, rw_u);
        set_critical_edge_index(rw_u, rw_v);

        filtered_edge_output(u, v, edge_filt(0));

        for (edge_idx_t idx = 1; idx < nb_edges; idx++) {
            u = edge_vertex(idx, filt_comp_t::u);
            v = edge_vertex(idx, filt_comp_t::v);

            // Inserts the edge in the sparse matrix to update the graph (G_i)
            insert_new_edge(u, v, idx);
            rw_u = adj_mat_col_idx(u);
            rw_v = adj_mat_col_idx(v);

            if (!edge_is_dominated(rw_u, rw_v, mask_critical_edge(idx))) {
                dominated_edge_idx[idx] = true;
                set_critical_edge_index(rw_v, rw_u);
                set_critical_edge_index(rw_u, rw_v);

                filtered_edge_output(u, v, edge_filt(idx));
                set_edge_critical(idx, edge_filt(idx), filtered_edge_output);
            }
        }
    }
};

/** \brief Implicitly constructs a flag complex from edges as an input,
 * collapses edges while preserving the persistent homology and returns the
 * remaining edges as a range.
 *
 * \param[in] edges Range of Filtered edges.There is no need the range to be
 * sorted, as it will be performed.
 *
 * \tparam FilteredEdgeRange furnishes `std::begin` and `std::end` methods and
 * returns an iterator on a FilteredEdge of type `std::tuple<vertex_t, vertex_t,
 * filt_t>` where `vertex_t` is the type of a vertex index and `filt_t` is the
 * type of an edge filtration value.
 *
 * \return Remaining edges after collapse as a range of
 * `std::tuple<vertex_t, vertex_t, filt_t>`.
 *
 * \ingroup edge_collapse
 *
 */
template <class FilteredEdgeRange>
auto flag_complex_collapse_edges(const FilteredEdgeRange& edges)
{
    using vertex_t = typename std::tuple_element<
        0, typename FilteredEdgeRange::value_type>::type;
    using filt_t = typename std::tuple_element<
        2, typename FilteredEdgeRange::value_type>::type;
    using edge_collapser_t = Flag_complex_edge_collapser<vertex_t, filt_t>;
    std::vector<typename edge_collapser_t::filt_edge_t> remaining_edges;

    /* Collapse only if at least 1 edge exist */
    if (edges.size()) {
        remaining_edges.reserve(edges.size());

        edge_collapser_t edge_collapser(edges);
        edge_collapser.process_edges(
            [&remaining_edges](vertex_t u, vertex_t v, filt_t filtration) {
                // insert the edge
                remaining_edges.emplace_back(u, v, filtration);
            });
    }
    return remaining_edges;
}

}  // namespace collapse
}  // namespace Gudhi

#endif  // FLAG_COMPLEX_EDGE_COLLAPSER_H_
