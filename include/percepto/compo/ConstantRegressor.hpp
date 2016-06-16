#pragma once

#include "percepto/compo/Parametric.hpp"
#include "percepto/compo/Interfaces.h"
#include <iostream>

namespace percepto
{

/** 
 * \brief A regressor that returns a constant output.
 */
class ConstantVectorRegressor
: public Source<VectorType>
{
public:

	typedef VectorType OutputType;
	typedef Source<VectorType> SourceType;

	ConstantVectorRegressor( unsigned int dim ) 
	: _dim( dim ), _params( nullptr ), _W( nullptr, 0 ) {}

	ConstantVectorRegressor( const ConstantVectorRegressor& other )
	: _dim( other._dim ), _params( other._params ), _W( other._W ) {}

	Parameters::Ptr CreateParameters()
	{
		Parameters::Ptr params = std::make_shared<Parameters>();
		params->Initialize( _dim );
		SetParameters( params );
		return params;
	}

	void SetParameters( Parameters::Ptr params )
	{
		_params = params;
		new (&_W) Eigen::Map<const VectorType>( params->GetParamsVec().data(), 
		                                        params->ParamDim() );
	}

	virtual void BackpropImplementation( const MatrixType& nextDodx )
	{
		if( nextDodx.size() > 0 )
		{
			// std::cout << "Constant: nextDodx: " << nextDodx << std::endl;
			_params->AccumulateDerivs( nextDodx );
		}
	}

	virtual void Foreprop()
	{
		SourceType::SetOutput( _W );
		SourceType::Foreprop();
	}

private:

	unsigned int _dim;
	Parameters::Ptr _params;
	Eigen::Map<const VectorType> _W;
};

}