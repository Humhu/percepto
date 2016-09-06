#pragma once

#include "modprop/compo/Interfaces.h"
#include "modprop/ModpropTypes.h"
#include <iostream>

namespace percepto
{

class ScalarProductWrapper
: public Source<double>
{
public:

	typedef Source<double> SourceType;
	typedef Sink<double> SinkType;

	ScalarProductWrapper()
	: _inputA( this ), _inputB( this ) {}

	ScalarProductWrapper( const ScalarProductWrapper& other )
	: _inputA( this ), _inputB( this ) {}

	void SetSourceA( SourceType* a ) { a->RegisterConsumer( &_inputA ); }
	void SetSourceB( SourceType* b ) { b->RegisterConsumer( &_inputB ); }

	virtual void Foreprop()
	{
		if( _inputA.IsValid() && _inputB.IsValid() )
		{
			SourceType::SetOutput( _inputA.GetInput() * _inputB.GetInput() );
			SourceType::Foreprop();
		}
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		_inputA.Backprop( nextDodx * _inputB.GetInput() );
		_inputB.Backprop( nextDodx * _inputA.GetInput() );
	}

private:

	SinkType _inputA;
	SinkType _inputB;
};

class VectorProductWrapper
: public Source<VectorType>
{
public:

	typedef Source<MatrixType> MatSourceType;
	typedef Source<VectorType> VecSourceType;
	typedef VectorType OutputType;

	VectorProductWrapper()
	: _mat( this ), _vec( this ) {}

	VectorProductWrapper( const VectorProductWrapper& other )
	: _mat( this ), _vec( this ) {}

	void SetMatSource( MatSourceType* s ) { s->RegisterConsumer( &_mat ); }
	void SetVecSource( VecSourceType* s ) { s->RegisterConsumer( &_vec ); }

	virtual void Foreprop()
	{
		if( _mat.IsValid() && _vec.IsValid() )
		{
			VecSourceType::SetOutput( _mat.GetInput() * _vec.GetInput() );
			VecSourceType::Foreprop();
		}
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{

		// clock_t start = clock();

		MatrixType M = _mat.GetInput();
		VectorType v = _vec.GetInput();
		
		if( nextDodx.cols() != M.rows() )
		{
			throw std::runtime_error( "VectorProductWrapper: Backprop dim error." );
		}


		MatrixType thisDodw = MatrixType( nextDodx.rows(), M.size() );
		for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		{
			for( unsigned int j = 0; j < M.rows(); j++ )
			{
				thisDodw.block(i, j*M.cols(), 1, M.cols()) =
					nextDodx(i,j) * v.transpose();
			}
		}
		// std::cout << "Product backprop: " << ((double) clock() - start)/CLOCKS_PER_SEC << std::endl;;

		_vec.Backprop( nextDodx * M );
		_mat.Backprop( thisDodw );
	}

private:

	Sink<MatrixType> _mat;
	Sink<VectorType> _vec;
};

class ProductWrapper
: public Source<MatrixType>
{
public:

	typedef Source<MatrixType> SourceType;
	typedef Sink<MatrixType> SinkType;
	typedef MatrixType OutputType;

	ProductWrapper() 
	: _left( this ), _right( this ) {}

	ProductWrapper( const ProductWrapper& other ) 
	: _left( this ), _right( this ) {}

	void SetLeftSource( SourceType* l ) { l->RegisterConsumer( &_left ); }
	void SetRightSource( SourceType* r ) { r->RegisterConsumer( &_right ); }

	virtual void Foreprop()
	{
		if( _left.IsValid() && _right.IsValid() )
		{
			SourceType::SetOutput( _left.GetInput() * _right.GetInput() );
			SourceType::Foreprop();
		}
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		const MatrixType& left = _left.GetInput();
		const MatrixType& right = _right.GetInput();
		unsigned int outputDim = left.rows() * right.cols();
		if( nextDodx.cols() != outputDim )
		{
			throw std::runtime_error( "ProductWrapper: Backprop dim error." );
		}

		MatrixType dSdL( outputDim, left.size() );
		// MatrixType dSdLo( outputDim, left.size() );
		for( unsigned int i = 0; i < left.rows(); i++ )
		{
			for( unsigned int j = 0; j < left.cols(); j++ )
			{
				MatrixType dSdi = MatrixType::Zero( left.rows(), right.cols() );
				dSdi.row(i) = right.row(j);
				unsigned int ind = i + j*left.rows();
				dSdL.col(ind) = Eigen::Map<VectorType>( dSdi.data(), dSdi.size() );
			}
		}

		// MatrixType d = MatrixType::Zero( left.rows(), left.cols() );
		// for( unsigned int i = 0; i < left.size(); i++ )
		// {
		// 	d(i) = 1;
		// 	MatrixType dSdi = d * right;
		// 	dSdLo.col(i) = Eigen::Map<VectorType>( dSdi.data(), dSdi.size() );
		// 	d(i) = 0;
		// }

		// if( ( (dSdLo - dSdL).array().abs() > 1E-3 ).any() )
		// {
		// 	throw std::runtime_error( "dSdL calculation wrong!" );
		// }

		MatrixType midLInfoDodx = nextDodx * dSdL;
		_left.Backprop( midLInfoDodx );

		MatrixType dSdR( outputDim, right.size() );
		// MatrixType dSdRo( outputDim, right.size() );
		for( unsigned int i = 0; i < right.rows(); i++ )
		{
			for( unsigned int j = 0; j < right.cols(); j++ )
			{
				MatrixType dSdi = MatrixType::Zero( left.rows(), right.cols() );
				dSdi.col(j) = left.col(i);
				unsigned int ind = i + j*right.rows();
				dSdR.col(ind) = Eigen::Map<VectorType>( dSdi.data(), dSdi.size() );
			}
		}

		// d = MatrixType::Zero( right.rows(), right.cols() );
		// for( unsigned int i = 0; i < right.size(); i++ )
		// {
		// 	d(i) = 1;
		// 	MatrixType dSdi = left * d;
		// 	dSdRo.col(i) = Eigen::Map<VectorType>( dSdi.data(), dSdi.size() );
		// 	d(i) = 0;
		// }

		// if( ( (dSdRo - dSdR).array().abs() > 1E-3 ).any() )
		// {
		// 	throw std::runtime_error( "dSdR calculation wrong!" );
		// }

		MatrixType midRInfoDodx = nextDodx * dSdR;
		_right.Backprop( midRInfoDodx );
	}

private:

	SinkType _left;
	SinkType _right;
};

}