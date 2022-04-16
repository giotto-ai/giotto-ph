/******************************************************************************
 * Author:           Julián Burella Pérez
 * Description:      Parallel Ripser Vietoris-Rips interfacing with pybind11
 * License:          AGPL3
 *****************************************************************************/

#include <ripser.h>

// PYBIND11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

/* This function allows a conversion from a vector of vector into a
 * vector of numpy arrays. This would be translated in Python into a
 * List[np.array, ...]
 */
std::vector<py::array_t<value_t>>
to_numpy_barcodes(std::vector<barcodes_t> barcodes)
{
    auto mat = std::vector<py::array_t<value_t>>(barcodes.size());

    size_t i = 0;
    for (auto& barcode : barcodes) {
        auto arr = py::array_t<value_t>(
            py::buffer_info(barcode.data(),
                            sizeof(value_t),  // itemsize
                            py::format_descriptor<value_t>::format(),
                            barcode.size()  // shape
                            ));

        mat[i++] = arr;
    }

    return mat;
}

#if defined USE_COEFFICIENTS
PYBIND11_MODULE(gph_ripser_coeff, m)
{
#else
PYBIND11_MODULE(gph_ripser, m)
{
#endif

    using namespace pybind11::literals;
    m.doc() = "Ripser python interface";

    py::class_<flagPersGen>(m, "flagPersGen", py::module_local())
        .def_readwrite("finite_0", &flagPersGen::finite_0)
        .def_readwrite("finite_higher", &flagPersGen::finite_higher)
        .def_readwrite("essential_0", &flagPersGen::essential_0)
        .def_readwrite("essential_higher", &flagPersGen::essential_higher);
    // Because `ripser` could have two different modules after compilation
    // It's necessary to add `py::module_local()` to prevent following issue:
    // ImportError: generic_type: type "ripserResults" is already registered!
    // When same python module imports gtda_ripser and gtda_ripser_coeff
    py::class_<ripserResults>(m, "ripserResults", py::module_local())
        .def("births_and_deaths_by_dim",
             [&](ripserResults& res) {
                 return to_numpy_barcodes(res.births_and_deaths_by_dim);
             })
        .def_readwrite("flag_persistence_generators_by_dim",
                       &ripserResults::flag_persistence_generators);

    m.def(
        "rips_dm",
        [](py::array_t<value_t>& D, py::array_t<value_t>& diag, int modulus,
           int dim_max, float threshold, int num_threads,
           bool return_generators) {
            // Setup distance matrix and figure out threshold
            auto D_ = static_cast<value_t*>(D.request().ptr);
            std::vector<value_t> distances(D_, D_ + D.size());
            auto diag_ = static_cast<value_t*>(diag.request().ptr);
            std::vector<value_t> diagonal(diag_, diag_ + diag.size());

            compressed_lower_distance_matrix dist =
                compressed_lower_distance_matrix(
                    compressed_upper_distance_matrix(std::move(distances),
                                                     std::move(diagonal)));

            ripserResults res;
            ripser<compressed_lower_distance_matrix> r(
                std::move(dist), dim_max, threshold, modulus, num_threads,
                return_generators);
            r.compute_barcodes();
            r.copy_results(res);
            return res;
        },
        "D"_a, "diag"_a, "modulus"_a, "dim_max"_a, "threshold"_a,
        "num_threads"_a, "return_generators"_a, "ripser distance matrix");

    m.def(
        "rips_dm_sparse",
        [](py::array_t<index_t>& I, py::array_t<index_t>& J,
           py::array_t<value_t>& V, int NEdges, int N, int modulus, int dim_max,
           float threshold, int num_threads, bool return_generators) {
            auto I_ = static_cast<index_t*>(I.request().ptr);
            auto J_ = static_cast<index_t*>(J.request().ptr);
            auto V_ = static_cast<value_t*>(V.request().ptr);

            // Setup distance matrix and figure out threshold
            ripser<sparse_distance_matrix> r(
                sparse_distance_matrix(I_, J_, V_, NEdges, N, threshold),
                dim_max, threshold, modulus, num_threads, return_generators);
            r.compute_barcodes();

            ripserResults res;
            r.copy_results(res);
            return res;
        },
        "I"_a, "J"_a, "V"_a, "NEdges"_a, "N"_a, "modulus"_a, "dim_max"_a,
        "threshold"_a, "num_threads"_a, "return_generators"_a,
        "ripser sparse distance matrix");

    m.def("get_max_coefficient_field_supported",
          []() { return (uintptr_t(1) << num_coefficient_bits) - 1; });
}
