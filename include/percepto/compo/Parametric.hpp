#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/utils/MatrixUtils.hpp"
#include <memory>
#include <boost/foreach.hpp>
#include <iostream>

namespace percepto
{

// Class of objects with parameters
// TODO Make this an interface so ParameterWrapper isn't so awkward
class Parameters
{
public:

	typedef std::shared_ptr<Parameters> Ptr;

	Parameters() {}

	void Initialize( unsigned int dim )
	{
		_params = VectorType( dim );
	}

	virtual void AccumulateDerivs( const MatrixType& delDodw )
	{
		if( delDodw.cols() != ParamDim() )
		{
			std::cout << "delDodw size: " << delDodw.size() << std::endl;
			std::cout << "ParamDim: " << ParamDim() << std::endl;
			throw std::runtime_error( "Parameters: Accumulation dim error." );
		}

		if( _dodw.size() == 0 ) { _dodw = delDodw; }
		else { _dodw += delDodw; }
	}

	virtual void ResetAccumulators()
	{
		_dodw = MatrixType();
	}

	virtual const MatrixType& GetDerivs() const
	{
		return _dodw;
	}

	virtual unsigned int ParamDim() const { return _params.size(); }
	
	virtual const VectorType& GetParamsVec() const { return _params; }
	
	virtual void SetParamsVec( const VectorType& vec )
	{
		if( vec.size() != _params.size() )
		{
			throw std::runtime_error( "Parameters: Cannot change size of parameters." );
		}
		_params = vec;
	}

protected:

	MatrixType _dodw;
	VectorType _params;
};

// Wraps a group of parameters
// Should not be treated as parameter memory as the local params
// copy is not updated
class ParameterWrapper
: public Parameters
{

public:

	typedef std::vector<Parameters::Ptr> ContainerType;
	typedef std::shared_ptr<ParameterWrapper> Ptr;

	ParameterWrapper() {}

	// TODO Remove this interface
	ParameterWrapper( const ContainerType& p ) 
	: _items( p )
	{
		Initialize( ParamDim() );
	}

	bool AddParameters( const Parameters::Ptr& p )
	{
		if( !p ) { return false; }
		BOOST_FOREACH( Parameters::Ptr& item, _items )
		{
			if( item == p ) { return false; }
		}
		_items.push_back( p );
		Initialize( ParamDim() );
		return true;
	}

	// This can get called by modules that operate directly on weights,
	// like ParameterL2Cost
	virtual void AccumulateDerivs( const MatrixType& dodw )
	{
		if( dodw.cols() != ParamDim() )
		{
			throw std::runtime_error( "ParametersWrapper: Accumulate dim error!" );
		}

		unsigned int ind = 0;
		BOOST_FOREACH( Parameters::Ptr& item, _items )
		{
			unsigned int n = item->ParamDim();
			item->AccumulateDerivs( dodw.block( 0, ind, dodw.rows(), n ) );
			ind += n;
		}
	}

	virtual void ResetAccumulators() 
	{
		_accDodw = MatrixType();
		BOOST_FOREACH( Parameters::Ptr& item, _items )
		{
			item->ResetAccumulators();
		}
	}

	virtual const MatrixType& GetDerivs() const
	{
		_accDodw = MatrixType();
		if( _items.empty() ) { return _accDodw; }

		// First determine the sys output dim by checking for the
		// first item with a valid dodw
		unsigned int sysOutDim = 0;
		BOOST_FOREACH( const Parameters::Ptr& item, _items )
		{
			if( item->GetDerivs().size() == 0 ) { continue; }
			sysOutDim = item->GetDerivs().rows();
			break;
		}
		if( sysOutDim == 0 ) { return _accDodw; }

		_accDodw = MatrixType( sysOutDim, ParamDim() );
		unsigned int ind = 0;
		BOOST_FOREACH( const Parameters::Ptr& item, _items )
		{
			unsigned int n = item->ParamDim();
			MatrixType itemDodw = item->GetDerivs();
			if( itemDodw.size() == 0 )
			{
				itemDodw = MatrixType::Zero( sysOutDim, n );
			}
			_accDodw.block( 0, ind, sysOutDim, n ) = itemDodw;
			ind += n;
		}
		return _accDodw;
	}

	virtual unsigned int ParamDim() const
	{
		unsigned int acc = 0;
		BOOST_FOREACH( const Parameters::Ptr& item, _items )
		{
			acc += item->ParamDim();
		}
		return acc;
	}

	virtual const VectorType& GetParamsVec() const
	{
		_accParams = VectorType( ParamDim() );
		unsigned int ind = 0;
		BOOST_FOREACH( const Parameters::Ptr& item, _items )
		{
			unsigned int n = item->ParamDim();
			_accParams.segment( ind, n ) = item->GetParamsVec();
			ind += n;
		}
		return _accParams;
	}

	virtual void SetParamsVec( const VectorType& vec )
	{
		if( vec.size() != ParamDim() )
		{
			throw std::runtime_error( "ParametersWrapper: Invalid params vec size in SetParamsVec()." );
		}
		unsigned int ind = 0;
		BOOST_FOREACH( Parameters::Ptr& item, _items )
		{
			unsigned int n = item->ParamDim();
			item->SetParamsVec( vec.segment( ind, n ) );
			ind += n;
		}
		_accParams = vec;
	}

private:

	mutable VectorType _accParams;
	mutable MatrixType _accDodw;
	std::vector<Parameters::Ptr> _items;

};

}