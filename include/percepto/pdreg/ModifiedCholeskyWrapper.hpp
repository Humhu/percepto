#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/BackpropInfo.hpp"
#include "percepto/pdreg/LowerTriangular.hpp"

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

	struct ParamType
	{
		typename LRegressorType::ParamType lParams;
		typename DRegressorType::ParamType dParams;
	};

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

	static ParamType create_zeros( unsigned int lInputDim,
	                               unsigned int dInputDim,
	                               unsigned int outputDim )
	{
		typedef TriangularMapping TMap;
		unsigned int lOutputDim = TMap::num_positions( outputDim - 1 );
		unsigned int dOutputDim = outputDim;

		ParamType p;
		p.lParams = LRegressorType::create_zeros( lInputDim,
		                                              lOutputDim );
		p.dParams = DRegressorType::create_zeros( dInputDim,
		                                              dOutputDim );
		return p;
	}

	/*! \brief Instantiate an estimator by giving it regressors for 
	 * the modified Cholesky predictors. Makes copies of the regressors. */
	ModifiedCholeskyWrapper( const LRegressorType& l, const DRegressorType& d, 
	                         const MatrixType& offset )
	: _lRegressor( l ), _dRegressor( d ), _tmap( _dRegressor.OutputDim() - 1 ),
	 _offset( offset )
	{
		InitCheckDimensions();
	}

	ModifiedCholeskyWrapper( const ParamType& p, 
	                         const MatrixType& offset )
	: _lRegressor( p.lParams ), _dRegressor( p.dParams ), 
	_tmap( _dRegressor.OutputDim() - 1 ), _offset( offset )
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

	void SetParams( const ParamType& p ) 
	{ 
		_lRegressor.SetParams( p.lParams );
		_dRegressor.SetParams( p.dParams );
	}

	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		_lRegressor.SetParamsVec( v.block( 0, 0, _lRegressor.ParamDim(), 1 ) );
		_dRegressor.SetParamsVec( v.block( _lRegressor.ParamDim(), 0, 
		                                   _dRegressor.ParamDim(), 1 ) );
	}

	/*! \brief Returns a deep copy of the current parameters. */
	ParamType GetParams() const 
	{ 
		ParamType p;
		p.lParams = _lRegressor.GetParams();
		p.dParams = _dRegressor.GetParams();
		return p;
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
	void Backprop( const InputType& input, const BackpropInfo& nextInfo,
	               BackpropInfo& thisInfo )
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
		BackpropInfo midLInfo, lInfo;
		midLInfo.dodx = nextInfo.dodx * dSdl;
		_lRegressor.Backprop( input.lInput, midLInfo, lInfo );

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
		BackpropInfo midDInfo, dInfo;
		midDInfo.dodx = nextInfo.dodx * dSdd;
		_dRegressor.Backprop( input.dInput, midDInfo, dInfo );

		thisInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), lInfo.ModuleInputDim() + dInfo.ModuleInputDim() );
		thisInfo.dodx << lInfo.dodx, dInfo.dodx;
		thisInfo.dodw = MatrixType( nextInfo.SystemOutputDim(), lInfo.ModuleParamDim() + dInfo.ModuleParamDim() );
		thisInfo.dodw << lInfo.dodw, dInfo.dodw;
	}

	/*! \brief Compute all derivatives, ordered in a vector. */
	std::vector<OutputType> AllDerivatives( const InputType& features ) const
	{
		std::vector<OutputType> lDerivatives = AllLDerivatives( features );
		std::vector<OutputType> dDerivatives = AllDDerivatives( features );
		lDerivatives.reserve( ParamDim() );
		lDerivatives.insert( lDerivatives.end(), dDerivatives.begin(), dDerivatives.end() );
		return lDerivatives;
	}

	OutputType Derivative( const InputType& features, unsigned int ind ) const
	{
		assert( ind < _lRegressor.ParamDim() + _dRegressor.ParamDim() );

		if( ind < _lRegressor.ParamDim() )
		{
			return LDerivative( features, ind );
		}
		return DDerivative( features, ind - _lRegressor.ParamDim() );

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

	// std::vector<OutputType> AllLDerivatives( const InputType& features ) const
	// {
	// 	std::vector<OutputType> derivs( _lRegressor.ParamDim() );

	// 	MatrixType& L = EvaluateL( features.lInput );
	// 	DiagonalType& D = EvaluateD( features.dInput );

	// 	// Diagonals will have 0 derivative
	// 	MatrixType dL = MatrixType::Zero( _L.rows(), _L.cols() );
	// 	for( unsigned int ind = 0; ind < _lRegressor.ParamDim(); ind++ )
	// 	{
	// 		_tmap.VecToLowerTriangular( _lRegressor.Derivative( features.lInput, ind ), dL );

	// 		MatrixType prod = dL * D * L.transpose();
	// 		derivs[ind] = prod + prod.transpose();
	// 	}
	// 	return derivs;
	// }


	// std::vector<OutputType> AllDDerivatives( const InputType& features ) const
	// {
	// 	std::vector<OutputType> derivs( _dRegressor.ParamDim() );

	// 	MatrixType& L = EvaluateL( features.lInput );
	// 	for( unsigned int ind = 0; ind < _dRegressor.ParamDim(); ind++ )
	// 	{
	// 		DiagonalType dD( _dRegressor.Derivative( features.dInput, ind ) );
	// 		derivs[ind] = L * dD * L.transpose();
	// 	}
	// 	return derivs;
	// }

	// /*! \brief Returns the matrix derivative with respect to the ind-th L
	//  * regressor parameter. */
	// OutputType LDerivative( const InputType& features, 
	//                         unsigned int ind ) const
	// {
	// 	MatrixType& L = EvaluateL( features.lInput );
	// 	DiagonalType& D = EvaluateD( features.dInput );

	// 	// Diagonals will have 0 derivative
	// 	MatrixType dL = MatrixType::Zero( _L.rows(), _L.cols() );
	// 	_tmap.VecToLowerTriangular( _lRegressor.Derivative( features.lInput, ind ), dL );

	// 	MatrixType prod = dL * D * L.transpose();
	// 	return prod + prod.transpose();
	// }

	// /*! \brief Returns the matrix derivative with respect to the ind-th D
	//  * regressor parameter. */
	// OutputType DDerivative( const InputType& features,
	//                         unsigned int ind ) const
	// {
	// 	MatrixType& L = EvaluateL( features.lInput );
	// 	DiagonalType dD( _dRegressor.Derivative( features.dInput, ind) );

	// 	return L * dD * L.transpose();
	// }

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

