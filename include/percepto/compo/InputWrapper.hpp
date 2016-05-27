#pragma once

namespace percepto
{

/**
 * \brief A wrapper that applies an input to the base.
 */
template <typename Regressor>
class InputWrapper
{
public:

	typedef Regressor RegressorType;
	typedef typename RegressorType::InputType HeldInputType;
	typedef typename RegressorType::OutputType OutputType;

	const HeldInputType input;

	InputWrapper( RegressorType& b, const HeldInputType& in )
	: input( in ), _base( b ){}

	unsigned int OutputDim() const { return _base.OutputDim(); }
	unsigned int ParamDim() const { return _base.ParamDim(); }

	void SetParamsVec( const VectorType& v ) { _base.SetParamsVec( v ); }
	VectorType GetParamsVec() const { return _base.GetParamsVec(); }

	BackpropInfo Backprop( const BackpropInfo& nextInfo )
	{
		return _base.Backprop( input, nextInfo );
	}

	OutputType Evaluate() const
	{
		return _base.Evaluate( input );
	}

private:

	RegressorType& _base;
};

}