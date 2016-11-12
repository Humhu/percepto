#include "covreg/AdaptiveCovarianceEstimator.h"
#include <boost/foreach.hpp>
#include "argus_utils/utils/ParamUtils.h"
#include "argus_utils/utils/MatrixUtils.h"

using namespace argus_msgs;
using namespace argus;

namespace percepto
{

AdaptiveTransitionCovarianceEstimator::AdaptiveTransitionCovarianceEstimator() 
: _lastDt( 1.0 ) {}

void AdaptiveTransitionCovarianceEstimator::Initialize( ros::NodeHandle& ph )
{
	unsigned int dim;
	GetParamRequired<unsigned int>( ph, "dim", dim );
	GetParamRequired( ph, "window_length", _windowLength );
	GetParam( ph, "use_diag", _useDiag, true );

	_initCov = MatrixType::Identity( dim, dim );
	if( !GetMatrixParam<double>( ph, "initial_cov", _initCov ) )
	{
		if( !GetDiagonalParam<double>( ph, "initial_cov", _initCov ) )
		{
			ROS_WARN_STREAM( "No initial covariance specified. Using identity.");
		}
	}

	_offset = 1E-9 * MatrixType::Identity( dim, dim );
	if( !GetMatrixParam<double>( ph, "offset", _offset ) )
	{
		if( !GetDiagonalParam<double>( ph, "offset", _offset ) )
		{
			ROS_WARN_STREAM( "No offset specified. Using 1E-9.");
		}
	}

	_currSpost = MatrixType::Zero( dim, dim );
	_lastFSpostFT = MatrixType::Zero( dim, dim );

	double decayRate;
	GetParam( ph, "decay_rate", decayRate, 1.0 );
	
	_prodWeights = VectorType( _windowLength );
	_prodWeights(0) = 1.0;
	for( unsigned int i = 1; i < _windowLength; ++i )
	{
		_prodWeights(i) = _prodWeights(i-1) * decayRate;
	}
	_prodWeights = _prodWeights / _prodWeights.sum();
}

MatrixType AdaptiveTransitionCovarianceEstimator::GetQ() const
{
	MatrixType acc = MatrixType::Zero( _currSpost.rows(), _currSpost.cols() );
	for( unsigned int i = 0; i < _delXOuterProds.size(); i++ )
	{
		acc += _delXOuterProds[i] * _prodWeights(i);
	}
	MatrixType adaptQ = acc + _currSpost + _lastFSpostFT + _offset;

	// Check for diagonal
	if( _useDiag )
	{
		adaptQ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>( adaptQ.diagonal() );
	}

	// Smoothly slew between adaptQ and initCov
	double adaptWeight = _delXOuterProds.size() / (double) _windowLength;
	double initWeight = 1.0 - adaptWeight;

	MatrixType ret = adaptWeight * adaptQ + initWeight * _initCov;
	return ret;
}

void AdaptiveTransitionCovarianceEstimator::Update( const PredictInfo& predict,
                                                    const UpdateInfo& update )
{
	// TODO This should really be time since last update
	MatrixType op = update.delta_x * update.delta_x.transpose() / predict.dt;
	_delXOuterProds.push_front( op );
	while( _delXOuterProds.size() > _windowLength )
	{
		_delXOuterProds.pop_back();
	}

	_lastFSpostFT = predict.F * _currSpost * predict.F.transpose();
	_currSpost = update.Spost;
}

void AdaptiveTransitionCovarianceEstimator::Reset()
{
	_delXOuterProds.clear();
}

AdaptiveObservationCovarianceEstimator::AdaptiveObservationCovarianceEstimator() {}

void AdaptiveObservationCovarianceEstimator::Initialize( ros::NodeHandle& ph )
{
	unsigned int dim;
	GetParamRequired<unsigned int>( ph, "dim", dim );
	GetParamRequired( ph, "window_length", _windowLength );
	GetParam( ph, "use_diag", _useDiag, true );

	_initCov = MatrixType::Identity( dim, dim );
	if( !GetMatrixParam<double>( ph, "initial_cov", _initCov ) )
	{
		if( !GetDiagonalParam<double>( ph, "initial_cov", _initCov ) )
		{
			ROS_WARN_STREAM( "No initial covariance specified. Using identity.");
		}
	}

	double decayRate;
	GetParam( ph, "decay_rate", decayRate, 1.0 );
	
	_prodWeights = VectorType( _windowLength );
	_prodWeights(0) = 1.0;
	for( unsigned int i = 1; i < _windowLength; ++i )
	{
		_prodWeights(i) = _prodWeights(i-1) * decayRate;
	}
	_prodWeights = _prodWeights / _prodWeights.sum();
}

MatrixType AdaptiveObservationCovarianceEstimator::GetR() const
{
	MatrixType acc = MatrixType::Zero( _lastHPHT.rows(), _lastHPHT.cols() );
	for( unsigned int i = 0; i < _innoOuterProds.size(); ++i )
	{
		acc += _innoOuterProds[i] * _prodWeights(i);
	}
	MatrixType adaptR = acc + _lastHPHT;
	
	// Check for diagonal
	if( _useDiag )
	{
		adaptR = Eigen::DiagonalMatrix<double, Eigen::Dynamic>( adaptR.diagonal() );
	}
	
	// Smoothly slew between initCov and adaptR
	double adaptWeight = _innoOuterProds.size() / (double) _windowLength;
	double initWeight = 1.0 - adaptWeight;
	return adaptWeight * adaptR + initWeight * _initCov;
}

void AdaptiveObservationCovarianceEstimator::Update( const UpdateInfo& update )
{
	// Update is R = Cv+ + H * P+ * H^
	_lastHPHT = update.H * update.Spost * update.H.transpose();
	MatrixType op = update.post_innovation * update.post_innovation.transpose();
	_innoOuterProds.push_front( op );

	// Remove old innovations
	while( _innoOuterProds.size() > _windowLength )
	{
		_innoOuterProds.pop_back();
	}
}

void AdaptiveObservationCovarianceEstimator::Reset()
{
	_innoOuterProds.clear();
}

}