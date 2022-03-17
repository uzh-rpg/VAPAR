#pragma once

#include <eigen3/Eigen/Eigen>

namespace flightlib {

// ------------ General Stuff-------------

// Define the scalar type used.
using Scalar = float;  // numpy float32

// Define frame id for unity
using FrameID = uint64_t;

// Define frame id for unity
using SceneID = size_t;


// ------------ Eigen Stuff-------------

// Define `Dynamic` matrix size.
static constexpr int Dynamic = Eigen::Dynamic;

// Using shorthand for `Matrix<rows, cols>` with scalar type.
template<int rows = Dynamic, int cols = Dynamic>
using Matrix = Eigen::Matrix<Scalar, rows, cols>;

// Using shorthand for `Matrix<rows, cols>` with scalar type.
template<int rows = Dynamic, int cols = Dynamic>
using MatrixRowMajor = Eigen::Matrix<Scalar, rows, cols, Eigen::RowMajor>;

// Using shorthand for `Vector<rows>` with scalar type.
template<int rows = Dynamic>
using Vector = Matrix<rows, 1>;

// Vector bool
template<int rows = Dynamic>
using BoolVector = Eigen::Matrix<bool, -1, 1>;

// Using shorthand for `Array<rows, cols>` with scalar type.
template<int rows = Dynamic, int cols = rows>
using Array = Eigen::Array<Scalar, rows, cols>;

// Using `SparseMatrix` with type.
using SparseMatrix = Eigen::SparseMatrix<Scalar>;

// Using SparseTriplet with type.
using SparseTriplet = Eigen::Triplet<Scalar>;

// Using `Quaternion` with type.
using Quaternion = Eigen::Quaternion<Scalar>;

// Using `Ref` for modifier references.
template<class Derived>
using Ref = Eigen::Ref<Derived>;

// // Using `ConstRef` for constant references.
// template<class Derived>
// using ConstRef = const Eigen::Ref<const Derived>;

// Using `Map`.
template<class Derived>
using Map = Eigen::Map<Derived>;


// ------------ Image Stuff------------

//
// template<int rows = Dynamic>
// using ImageFlat = Eigen::Matrix<uint8_t, rows, 1>;
// using ImageFlat = Eigen::Matrix<uint8_t, Dynamic, 1>;

//
// template<int rows = Dynamic, int cols = Dynamic>
// using ImageChannel = Eigen::Matrix<uint8_t, rows, cols, Eigen::RowMajor>;


// ------------ Optical Flow Stuff------------

//
template<int rows = Dynamic>
using OpticalFlowFlat = Eigen::Matrix<float_t, rows, 1>;
// template<int cols = Dynamic>
// using OpticalFlowFlat = Eigen::Matrix<float_t, 1, cols>;
// using ImageFlat = Eigen::Matrix<uint8_t, Dynamic, 1>;

//
template<int rows = Dynamic, int cols = Dynamic>
using OpticalFlowChannel = Eigen::Matrix<float_t, rows, cols, Eigen::RowMajor>;
// using OpticalFlowChannel = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// using OpticalFlowChannel = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;
// using OpticalFlowChannel = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic>;


// ------------ Unified Stuff------------

//
template<typename tn = uint8_t, int rows = Dynamic>
using ImageFlat = Eigen::Matrix<tn, rows, 1>;
// using ImageFlat = Eigen::Matrix<uint8_t, Dynamic, 1>;

//
template<typename tn = uint8_t, int rows = Dynamic, int cols = Dynamic>
using ImageChannel = Eigen::Matrix<tn, rows, cols, Eigen::RowMajor>;


static constexpr Scalar Gz = -9.81;
const Vector<3> GVEC{0.0, 0.0, Gz};

}  // namespace flightlib
