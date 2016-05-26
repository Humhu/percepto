#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/compo/BackpropInfo.hpp"
#include "percepto/utils/LowerTriangular.hpp"

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

	/*! \brief Create a vector of weights corresponding to this
	 * regressor's L and D regression parameters. */
	VectorType CreateWeightVector( double lWeight, double dWeight ) const
	{
		unsigned int lSize = GetLRegressor().ParamDim();
		unsigned int dSize = GetDRegressor().ParamDim();
		VectorType weights( ParamDim() );
		weights << lWeight * VectorType::Ones(lSize),
		           dWeight * VectorType::Ones(dSize);
		return weights;
	}

	// static ParamType create_zeros( unsigned int lInputDim,
	//                                unsigned int dInputDim,
	//                                unsigned int outputDim )
	// {
	// 	typedef TriangularMapping TMap;
	// 	unsigned int lOutputDim = TMap::num_positions( outputDim - 1 );
	// 	unsigned int dOutputDim = outputDim;

	// 	ParamType p;
	// 	p.lParams = LRegressorType::create_zeros( lInputDim,
	// 	                                          lOutputDim );
	// 	p.dParams = DRegressorType::create_zeros( dInputDim,
	// 	                                          dOutputDim );
	// 	return p;
	// }

	/*! \brief Instantiate an estimator by giving it regressors for 
	 * the modified Cholesky predictors. Makes copies of the regressors. */
	ModifiedCholeskyWrapper( const LRegressorType& l, const DRegressorType& d, 
	                         const MatrixType& offset )
	: _lRegressor( l ), _dRegressor( d ), _tmap( _dRegressor.OutputDim() - 1 ),
	 _offset( offset )
	{
		InitCheckDimensions();
	}

	LRegressorType& GetLRegressor() { return _lRegressor; }
	DRegressorType& GetDRegressor() { return _dRegressor; }
	const LRegressorType& GetLRegressor() const { return _lRegressor; }
	const DRegressorType& GetDRegressor() const { return _dRegressor; }

	// NOTE We allow direct access to the offset since it is not a parameter
	MatrixType& Offset() { return _offset; }
	const MatrixType& Offset() const { return _offset; }

	std::pair<unsigned int, unsigned int> OutputSize() const
	{
		return std::pair<unsigned int, unsigned int>( _dRegressor.OutputDim(), 
		                                              _dRegressor.OutputDim() );
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
		_lRegressor.SetParamsVec( v.block( 0, 0, _lRegressor.ParamDim(), 1 ) );
		_dRegressor.SetParamsVec( v.block( _lRegressor.ParamDim(), 0, 
		                                   _dRegressor.ParamDim(), 1 ) );
	}

	VectorType GetParamsVec() const
	{
		VectorType vec( ParamDim() );
		vec.block( 0, 0, _lRegressor.ParamDim(), 1 ) = _lRegressor.GetParamsVec();
		vec.block( _lRegressor.ParamDim(), 0, 
		           _dRegressor.ParamDim(), 1 ) = _dRegressor.GetParamsVec();
		return vec;
	}

	// Assuming that dodx is given w.r.t. matrix col-major ordering
	BackpropInfo Backprop( const InputType& input, const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		MatrixType D = EvaluateD( input );
		MatrixType L = EvaluateL( input );

		// Calculate output matrix deriv wrt L inputs
		MatrixType d = MatrixType::Zero( OutputSize().first, OutputSize().second );
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
		MatrixType dSdd = MatrixType( OutputDim(), OutputSize().first );
		for( unsigned int i = 0; i < OutputSize().first; i++ )
		{
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
		thisInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), lInfo.ModuleInputDim() + dInfo.ModuleInputDim() );
		thisInfo.dodx << lInfo.dodx, dInfo.dodx;
		thisInfo.dodw = MatrixType( nextInfo.SystemOutputDim(), lInfo.ModuleParamDim() + dInfo.ModuleParamDim() );
		thisInfo.dodw << lInfo.dodw, dInfo.dodw;
		return thisInfo;
	}

	OutputType Evaluate( const InputType& features ) const
	{
		MatrixType& L = EvaluateL( features );
		DiagonalType& D = EvaluateD( features );
		return L * D * L.transpose() + _offset;
	}
	
private:
	
	LRegressorType _lRegressor;
	DRegressorType _dRegressor;
	TriangularMapping _tmap;
	MatrixType _offset;

	typedef Eigen::DiagonalMatrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> DiagonalType;

	MatrixType& EvaluateL( const InputType& input ) const
	{
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
		assert( _offset.rows() == _dRegressor.OutputDim() );
		assert( _offset.cols() == _dRegressor.OutputDim() );

		_L =  MatrixType::Identity( OutputSize().first, OutputSize().second );
		_D = DiagonalType( OutputDim() );
		_D.setZero();
	}

	static void TypeTest()
	{
		typedef std::is_same<typename LRegressorType::ScalarType,
		                     typename DRegressorType::ScalarType> Same;
		static_assert( Same::value, "L and D regressor precision must be the same." );
	}

	// Workspace
	mutable MatrixType _L;
	mutable DiagonalType _D;

};

}

