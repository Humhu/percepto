#pragma once

#include "percepto/compo/Interfaces.h"
#include "percepto/PerceptoTypes.h"

namespace percepto
{

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
		// std::cout << "ProductWrapper backprop" << std::endl;
		const MatrixType& left = _left.GetInput();
		const MatrixType& right = _right.GetInput();
		unsigned int outputDim = left.rows() * right.cols();

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