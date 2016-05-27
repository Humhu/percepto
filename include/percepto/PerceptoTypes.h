#pragma once

#include <Eigen/Dense>

namespace percepto 
{

typedef double ScalarType;
typedef Eigen::MatrixXd MatrixType;
typedef Eigen::VectorXd VectorType;

typedef Eigen::Map<MatrixType> MatrixViewType;
typedef Eigen::Map<const MatrixType> ConstMatrixViewType;
typedef Eigen::Map<VectorType> VectorViewType;
typedef Eigen::Map<const VectorType> ConstVectorViewType;

struct MatrixSize
{
	size_t rows;
	size_t cols;

	MatrixSize( size_t r, size_t c )
	: rows( r ), cols( c ) {}
};

}
