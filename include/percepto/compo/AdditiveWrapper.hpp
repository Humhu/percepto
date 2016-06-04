#pragma once

#include "percepto/PerceptoTypes.h"
#include "percepto/utils/MatrixUtils.hpp"

namespace percepto
{

/*! \brief Adds two non-regressor bases together. */
template <typename BaseA, typename BaseB>
class AdditiveWrapper
{
public:

	typedef BaseA BaseAType;
	typedef BaseB BaseBType;
	typedef typename BaseAType::OutputType OutputType;

	AdditiveWrapper( BaseAType& baseA, BaseBType& baseB )
	: _baseA( baseA ), _baseB( baseB )
	{}

	MatrixSize OutputSize() const { return _baseA.OutputSize(); }
	unsigned int OutputDim() const { return _baseA.OutputDim(); }
	unsigned int ParamDim() const { return _baseA.ParamDim() + _baseB.ParamDim(); }

	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		_baseA.SetParamsVec( v.head( _baseA.ParamDim() ) );
		_baseB.SetParamsVec( v.tail( _baseB.ParamDim() ) );
	}

	VectorType GetParamsVec() const
	{
		return ConcatenateVer( _baseA.GetParamsVec(), _baseB.GetParamsVec() );
	}

	OutputType Evaluate() const
	{
		return _baseA.Evaluate() + _baseB.Evaluate();
	}

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo accInfo; // We are going to concatenate all infos
		accInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), 2 * OutputDim() );
		accInfo.dodw = MatrixType( nextInfo.SystemOutputDim(), ParamDim() );
		
		BackpropInfo baseInfo = _baseA.Backprop( nextInfo );
		accInfo.dodx.leftCols( _baseA.OutputDim() ) = baseInfo.dodx;
		accInfo.dodw.leftCols( _baseA.ParamDim() ) = baseInfo.dodw;
		baseInfo = _baseB.Backprop( nextInfo );
		accInfo.dodx.rightCols( _baseB.OutputDim() ) = baseInfo.dodx;
		accInfo.dodw.rightCols( _baseB.ParamDim() ) = baseInfo.dodw;
		return accInfo;
	}

private:

	BaseAType& _baseA;
	BaseBType& _baseB;
};

/*! \brief Adds a number of non-regressor bases together. */
template <typename Base, template<typename,typename> class Container = std::vector>
class AdditiveSumWrapper
{
public:

	typedef Base BaseType;
	typedef Container<Base, std::allocator<Base>> ContainerType;
	typedef typename BaseType::OutputType OutputType;

	AdditiveSumWrapper( ContainerType& bases )
	: _bases( bases ) {}

	MatrixSize OutputSize() const { return _bases[0].OutputSize(); }
	unsigned int OutputDim() const { return _bases[0].OutputDim(); }
	unsigned int ParamDim() const 
	{ 
		unsigned int acc = 0;
		for( unsigned int i = 0; i < _bases.size(); ++i )
		{
			acc += _bases[i].ParamDim();
		}
		return acc;
	}

	void SetParamsVec( const VectorType& v )
	{
		assert( v.size() == ParamDim() );
		unsigned int ind = 0;
		for( unsigned int i = 0; i < _bases.size(); ++i )
		{
			_bases[i].SetParamsVec( v.segment( ind, _bases[i].ParamDim() ) );
			ind += _bases[i].ParamDim();
		}
	}

	VectorType GetParamsVec() const
	{
		VectorType v( ParamDim() );
		unsigned int ind = 0;
		for( unsigned int i = 0; i < _bases.size(); ++i )
		{
			v.segment( ind, _bases[i].ParamDim() ) = _bases[i].GetParamsVec();
			ind += _bases[i].ParamDim();
		}
		return v;
	}

	OutputType Evaluate() const
	{
		OutputType out = _bases[0].Evaluate();
		for( unsigned int i = 1; i < _bases.size(); ++i )
		{
			out += _bases[i].Evaluate();
		}
		return out;
	}

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		assert( nextInfo.ModuleInputDim() == OutputDim() );

		BackpropInfo accInfo; // We are going to concatenate all infos
		unsigned int sumInputDim = _bases.size() * OutputDim();
		accInfo.dodx = MatrixType( nextInfo.SystemOutputDim(), sumInputDim );
		accInfo.dodw = MatrixType( nextInfo.SystemOutputDim(), ParamDim() );
		
		unsigned int xInd = 0, wInd = 0;
		for( unsigned int i = 0; i < _bases.size(); ++i )
		{
			BackpropInfo baseInfo = _bases[i].Backprop( nextInfo );
			accInfo.dodx.block( 0, xInd, 0, _bases[i].OutputDim() ) = baseInfo.dodx;
			accInfo.dodw.block( 0, wInd, 0, _bases[i].ParamDim() ) = baseInfo.dodw;
			xInd += _bases[i].OutputDim();
			wInd += _bases[i].ParamDim();
		}
		return accInfo;
	}

private:

	ContainerType& _bases;
};

}