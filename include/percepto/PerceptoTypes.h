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

}
