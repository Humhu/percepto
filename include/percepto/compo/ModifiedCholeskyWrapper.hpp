#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/compo/BackpropInfo.hpp"
#include "percepto/utils/LowerTriangular.hpp"
#include "percepto/utils/MatrixUtils.hpp"

namespace percepto
{

/*! \brief Positive-definite matrix regressor that regresses the L and D matrices
 * of a modified Cholesky decomposition and reforms them into a matrix. Uses
 * a different regressor for the L and D terms, but the same features. Orders
 * concatenated parameters with L parameters first, then D. */
template <typename LRegressor, typename DRegressor>
class ModifiedCholeskyWrapper
{
public:

	typedef LRegressor LRegressorType;
	typedef DRegressor DRegressorType;

	struct InputType
	{
		typename LRegressorType::InputType lInput;
		typename DRegressorType::InputType dInput;
	};

	typedef MatrixType OutputType;

	/*! \brief Instantiate an estimator by giving it regressors for 
	 * the modified Cholesky predictors. Makes copies of the regressors. */
	ModifiedCholeskyWrapper( const LRegressorType& l, const DRegressorType& d )
	: _lRegressor( l ), _dRegressor( d ), _tmap( _dRegressor.OutputDim() - 1 )
	{
		InitCheckDimensions();
	}

	MatrixSize OutputSize() const
	{
		return MatrixSize( _dRegressor.OutputDim(), _dRegressor.OutputDim() );
	}
	unsigned int OutputDim() const 
	{ 
		return _dRegressor.OutputDim() * _dRegressor.OutputDim(); 
	}
	
	unsigned int ParamDim() const 
	{ 
		return _lRegressor.ParamDim() + _dRegressor.ParamDim(); 
	}


	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		_lRegressor.SetParamsVec( v.head( _lRegressor.ParamDim() ) );
		_dRegressor.SetParamsVec( v.head( _dRegressor.ParamDim() ) );
	}

	VectorType GetParamsVec() const
	{
		VectorType vec( ParamDim() );
		vec << _lRegressor.GetParamsVec(), _dRegressor.GetParamsVec();
		return vec;
	}

	// Assuming that dodx is given w.r.t. matrix col-major ordering
	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		MatrixType D = EvaluateD( input );
		MatrixType L = EvaluateL( input );

		// Calculate output matrix deriv wrt L inputs
		MatrixType d = MatrixType::Zero( OutputSize().rows, OutputSize().cols );
		MatrixType dSdl = MatrixType( OutputDim(), _tmap.NumPositions() );
		for( unsigned int i = 0; i < _tmap.NumPositions(); i++ )
		{
			const TriangularMapping::Index& ind = _tmap.PosToIndex( i );
			// Have to add one to get the offset from the diagonal
			d( ind.first + 1, ind.second ) = 1;
			MatrixType dSdi = d * D * L.transpose() + L * D * d.transpose();
			Eigen::Map<VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
			dSdl.col(i) = dSdiVec;
			d( ind.first + 1, ind.second ) = 0;
		}
		BackpropInfo midLInfo;
		midLInfo.dodx = nextInfo.dodx * dSdl;
		BackpropInfo lInfo = _lRegressor.Backprop( input.lInput, midLInfo );

		// Perform D backprop
		MatrixType dSdd = MatrixType( OutputDim(), OutputSize().rows );
		for( unsigned int i = 0; i < OutputSize().rows; i++ )
		{
			// TODO Make more efficient with row product
			d( i, i ) = 1;
			MatrixType dSdi = L * d * L.transpose();
			Eigen::Map<VectorType> dSdiVec( dSdi.data(), dSdi.size(), 1 );
			dSdd.col(i) = dSdiVec;
			d( i, i ) = 0;
		}
		BackpropInfo midDInfo;
		midDInfo.dodx = nextInfo.dodx * dSdd;
		BackpropInfo dInfo = _dRegressor.Backprop( input.dInput, midDInfo );

		BackpropInfo thisInfo;
		thisInfo.dodx = ConcatenateHor( lInfo.dodx, dInfo.dodx );
		thisInfo.dodw = ConcatenateHor( lInfo.dodw, dInfo.dodw );
		return thisInfo;
	}

	OutputType Evaluate( const InputType& features ) const
	{
		MatrixType& L = EvaluateL( features );
		DiagonalType& D = EvaluateD( features );
		return L * D * L.transpose();
	}
	
private:
	
	LRegressorType _lRegressor;
	DRegressorType _dRegressor;
	TriangularMapping _tmap;

	typedef Eigen::DiagonalMatrix <ScalarType, 
	                               Eigen::Dynamic, 
	                               Eigen::Dynamic> DiagonalType;

	MatrixType& EvaluateL( const InputType& input ) const
	{
		MatrixType l = _lRegressor.Evaluate( input.lInput );
		_tmap.VecToLowerTriangular( _lRegressor.Evaluate( input.lInput ), _L );
		return _L;
	}

	DiagonalType& EvaluateD( const InputType& input ) const
	{
		_D.diagonal() = _dRegressor.Evaluate( input.dInput );
		return _D;
	}

	void InitCheckDimensions()
	{
		assert( _lRegressor.OutputDim() == _tmap.NumPositions() );

		_L =  MatrixType::Identity( OutputSize().rows, OutputSize().cols );
		_D = DiagonalType( _dRegressor.OutputDim() );
		_D.setZero();
	}

	// Workspace
	mutable MatrixType _L;
	mutable DiagonalType _D;

};

}

