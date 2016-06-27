#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"
#include <iostream>

namespace percepto
{

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

		MatrixType M = _mat.GetInput();
		VectorType v = _vec.GetInput();
		
		if( nextDodx.cols() != M.rows() )
		{
			throw std::runtime_error( "VectorProductWrapper: Backprop dim error." );
		}

		_vec.Backprop( nextDodx * M );

		MatrixType thisDodw = MatrixType( nextDodx.rows(), M.size() );
		for( unsigned int i = 0; i < nextDodx.rows(); i++ )
		{
			for( unsigned int j = 0; j < M.rows(); j++ )
			{
				thisDodw.block(i, j*M.cols(), 1, M.cols()) =
					nextDodx(i,j) * v.transpose();
			}
		}
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
		MatrixType d = MatrixType::Zero( left.rows(), left.cols() );
		for( unsigned int i = 0; i < left.size(); i++ )
		{
			d(i) = 1;
			MatrixType dSdi = d * right;
			dSdL.col(i) = Eigen::Map<VectorType>( dSdi.data(), dSdi.size() );
			d(i) = 0;
		}
		MatrixType midLInfoDodx = nextDodx * dSdL;
		// std::cout << "Product: midL: " << midLInfoDodx << std::endl;
		_left.Backprop( midLInfoDodx );

		MatrixType dSdR( outputDim, right.size() );
		d = MatrixType::Zero( right.rows(), right.cols() );
		for( unsigned int i = 0; i < right.size(); i++ )
		{
			d(i) = 1;
			MatrixType dSdi = left * d;
			dSdR.col(i) = Eigen::Map<VectorType>( dSdi.data(), dSdi.size() );
			d(i) = 0;
		}
		MatrixType midRInfoDodx = nextDodx * dSdR;
		// std::cout << "Product: midR: " << midRInfoDodx << std::endl;
		_right.Backprop( midRInfoDodx );
	}

private:

	SinkType _left;
	SinkType _right;
};

}