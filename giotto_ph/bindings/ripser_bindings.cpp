/******************************************************************************
 * Author:           Julián Burella Pérez
 * Description:      ripser's rips persistence interfacing with pybind11
 * License:          TBD
 *****************************************************************************/

#include <ripser.h>

// PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#if defined USE_COEFFICIENTS
PYBIND11_MODULE(giotto_ph_ripser_coeff, m)
{
#else
PYBIND11_MODULE(giotto_ph_ripser, m)
{
#endif

    using namespace pybind11::literals;
    m.doc() = "Ripser python interface";

    // Because `ripser` could have two different modules after compilation
    // It's necessary to add `py::module_local()` to prevent following issue:
    // ImportError: generic_type: type "ripserResults" is already registered!
    // When same python module imports gtda_ripser and gtda_ripser_coeff
    py::class_<ripserResults>(m, "ripserResults", py::module_local())
        .def_readwrite("births_and_deaths_by_dim",
                       &ripserResults::births_and_deaths_by_dim)
        .def_readwrite("num_edges", &ripserResults::num_edges);

    m.def(
        "rips_dm",
        [](py::array_t<value_t>& D, int N, int modulus, int dim_max,
           float threshold, int num_threads) {
            // Setup distance matrix and figure out threshold
            auto D_ = static_cast<value_t*>(D.request().ptr);
            std::vector<value_t> distances(D_, D_ + N);
            compressed_lower_distance_matrix dist =
                compressed_lower_distance_matrix(
                    compressed_upper_distance_matrix(std::move(distances)));

            // TODO: This seems like a dummy parameter at the moment
            float ratio = 1.0;

            value_t min = std::numeric_limits<value_t>::infinity(),
                    max = -std::numeric_limits<value_t>::infinity(),
                    max_finite = max;
            int num_edges = 0;

            for (auto d : dist.distances) {
                min = std::min(min, d);
                max = std::max(max, d);
                if (d != std::numeric_limits<value_t>::infinity())
                    max_finite = std::max(max_finite, d);
                if (d <= threshold)
                    ++num_edges;
            }

            if (threshold == std::numeric_limits<value_t>::infinity()) {
                threshold = std::numeric_limits<value_t>::max();
            }

            value_t enclosing_radius = std::numeric_limits<value_t>::infinity();
            if (threshold == std::numeric_limits<value_t>::max()) {
                for (size_t i = 0; i < dist.size(); ++i) {
                    value_t r_i = dist(i, 0);
                    for (size_t j = 1; j < dist.size(); ++j)
                        r_i = std::max(r_i, dist(i, j));
                    enclosing_radius = std::min(enclosing_radius, r_i);
                }

                if (enclosing_radius ==
                    std::numeric_limits<value_t>::infinity())
                    threshold = max_finite;
            }

            ripserResults res;
            if (threshold == std::numeric_limits<value_t>::max()) {
                ripser<compressed_lower_distance_matrix> r(
                    std::move(dist), dim_max, enclosing_radius, ratio, modulus,
                    num_threads);
                r.compute_barcodes();
                r.copy_results(res);
            } else {
                ripser<sparse_distance_matrix> r(
                    sparse_distance_matrix(std::move(dist), threshold), dim_max,
                    threshold, ratio, modulus, num_threads);
                r.compute_barcodes();
                r.copy_results(res);
            }
            res.num_edges = num_edges;
            return res;
        },
        "D"_a, "N"_a, "modulus"_a, "dim_max"_a, "threshold"_a, "num_threads"_a,
        "ripser distance matrix");

    m.def(
        "rips_dm_sparse",
        [](py::array_t<index_t>& I, py::array_t<index_t>& J,
           py::array_t<value_t>& V, int NEdges, int N, int modulus, int dim_max,
           float threshold, int num_threads) {
            auto I_ = static_cast<index_t*>(I.request().ptr);
            auto J_ = static_cast<index_t*>(J.request().ptr);
            auto V_ = static_cast<value_t*>(V.request().ptr);
            // TODO: This seems like a dummy parameter at the moment
            const float ratio = 1.0;
            // Setup distance matrix and figure out threshold
            ripser<sparse_distance_matrix> r(
                sparse_distance_matrix(I_, J_, V_, NEdges, N, threshold),
                dim_max, threshold, ratio, modulus, num_threads);
            r.compute_barcodes();
            // Report the number of edges that were added
            int num_edges = 0;
            for (int idx = 0; idx < NEdges; idx++) {
                if (I_[idx] < J_[idx] && V_[idx] <= threshold) {
                    num_edges++;
                }
            }
            ripserResults res;
            r.copy_results(res);
            res.num_edges = num_edges;
            return res;
        },
        "I"_a, "J"_a, "V"_a, "NEdges"_a, "N"_a, "modulus"_a, "dim_max"_a,
        "threshold"_a, "num_threads"_a, "ripser sparse distance matrix");
}
