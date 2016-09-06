#pragma once

#include <Eigen/Dense>
#include <stdexcept>

namespace percepto
{

template <class Derived>
Derived ConcatenateHor( const Eigen::DenseBase<Derived>& l,
                        const Eigen::DenseBase<Derived>& r )
{
	if( l.size() == 0 )
	{
		return r;
	}
	if( r.size() == 0 )
	{
		return l;
	}

	if( l.rows() != r.rows() )
	{
		throw std::runtime_error( "ConcatenateHor: Dimension mismatch." );
	}
	Derived out( l.rows(), l.cols() + r.cols() );
	out.leftCols( l.cols() ) = l;
	out.rightCols( r.cols() ) = r;
	return out;
}

template <class Derived>
Derived ConcatenateVer( const Eigen::DenseBase<Derived>& l,
                        const Eigen::DenseBase<Derived>& r )
{
	if( l.size() == 0 )
	{
		return r;
	}
	if( r.size() == 0 )
	{
		return l;
	}

	if( l.cols() != r.cols() )
	{
		throw std::runtime_error( "ConcatenateVer: Dimension mismatch." );
	}
	Derived out( l.rows() + r.rows(), l.cols() );
	out.topRows( l.rows() ) = l;
	out.bottomRows( r.rows() ) = r;
	return out;
}

}