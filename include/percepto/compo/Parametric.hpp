#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/utils/MatrixUtils.hpp"
#include <memory>
#include <boost/foreach.hpp>
#include <iostream>
namespace percepto
{

// Class of objects with parameters
class Parametric
{
public:

	Parametric() {}

	virtual void AccumulateWeightDerivs( const MatrixType& delDodw )
	{
		if( delDodw.cols() != ParamDim() )
		{
			throw std::runtime_error( "Parametric: Accumulation dim error." );
		}

		if( _dodw.size() == 0 ) { _dodw = delDodw; }
		else { _dodw += delDodw; }
	}

	virtual void ResetAccumulators()
	{
		_dodw = MatrixType();
	}

	virtual const MatrixType& GetAccWeightDerivs() const
	{
		return _dodw;
	}

	virtual unsigned int ParamDim() const = 0;
	virtual VectorType GetParamsVec() const = 0;
	virtual void SetParamsVec( const VectorType& vec ) = 0;

// private:

	MatrixType _dodw;

};

class ParametricWrapper
: public Parametric
{

public:

	typedef std::vector<Parametric*> ContainerType;

	ParametricWrapper() {}

	bool AddParametric( Parametric* p )
	{
		if( !p ) { return false; }
		BOOST_FOREACH( Parametric* item, _items )
		{
			if( item == p ) { return false; }
		}
		_items.push_back( p );
		return true;
	}

	// This can get called by modules that operate directly  on weights,
	// like ParameterL2Cost
	virtual void AccumulateWeightDerivs( const MatrixType& dodw )
	{
		if( dodw.cols() != ParamDim() )
		{
			throw std::runtime_error( "ParametricWrapper: Accumulate dim error!" );
		}

		unsigned int ind = 0;
		BOOST_FOREACH( Parametric* item, _items )
		{
			unsigned int n = item->ParamDim();
			item->AccumulateWeightDerivs( dodw.block( 0, ind, dodw.rows(), n ) );
			ind += n;
		}
	}

	virtual void ResetAccumulators() 
	{
		_accDodw = MatrixType();
		BOOST_FOREACH( Parametric* item, _items )
		{
			item->ResetAccumulators();
		}
	}

	virtual const MatrixType& GetAccWeightDerivs() const
	{
		_accDodw = MatrixType();
		if( _items.empty() ) { return _accDodw; }

		// First determine the sys output dim by checking for the
		// first item with a valid dodw
		unsigned int sysOutDim = 0;
		BOOST_FOREACH( const Parametric* item, _items )
		{
			if( item->GetAccWeightDerivs().size() == 0 ) { continue; }
			sysOutDim = item->GetAccWeightDerivs().rows();
			break;
		}
		if( sysOutDim == 0 ) { return _accDodw; }

		_accDodw = MatrixType( sysOutDim, ParamDim() );
		unsigned int ind = 0;
		BOOST_FOREACH( const Parametric* item, _items )
		{
			unsigned int n = item->ParamDim();
			MatrixType itemDodw = item->GetAccWeightDerivs();
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
		BOOST_FOREACH( const Parametric* item, _items )
		{
			acc += item->ParamDim();
		}
		return acc;
	}

	virtual VectorType GetParamsVec() const
	{
		VectorType params( ParamDim() );
		unsigned int ind = 0;
		BOOST_FOREACH( const Parametric* item, _items )
		{
			unsigned int n = item->ParamDim();
			params.segment( ind, n ) = item->GetParamsVec();
			ind += n;
		}
		return params;
	}

	virtual void SetParamsVec( const VectorType& vec )
	{
		if( vec.size() != ParamDim() )
		{
			throw std::runtime_error( "ParametricWrapper: Invalid params vec size in SetParamsVec()." );
		}
		unsigned int ind = 0;
		BOOST_FOREACH( Parametric* item, _items )
		{
			unsigned int n = item->ParamDim();
			item->SetParamsVec( vec.segment( ind, n ) );
			ind += n;
		}
	}

private:

	mutable MatrixType _accDodw;
	std::vector<Parametric*> _items;

};

}