#pragma once

namespace percepto
{

// Joins two networks together into a single network object
template <typename Head, typename Tail>
class NetWrapper
{
public:

	typedef Head HeadType;
	typedef typename HeadType::InputType InputType;
	typedef  Tail TailType;
	typedef typename TailType::OutputType OutputType;

	struct ParamType
	{
		typename HeadType::ParamType headParams;
		typename TailType::ParamType tailParams;
	};

	NetWrapper( HeadType& head, TailType& tail )
	: _head( head ), _tail( tail ) 
	{
		assert( head.OutputDim() == tail.InputDim() );
	}

	OutputType Evaluate( const InputType& input ) const
	{
		return _tail.Evaluate( _head.Evaluate( input ) );
	}

	BackpropInfo Backprop( const InputType& input,
	                      const BackpropInfo& nextNets ) const
	{
		BackpropInfo tailInfo = _tail.Backprop( _head.Evaluate( input ), nextNets );
		BackpropInfo headInfo = _head.Backprop( input, tailInfo );
		headInfo.dodw.conservativeResize( Eigen::NoChange, ParamDim() );
		headInfo.dodw.rightCols( _tail.ParamDim() ) = tailInfo.dodw;
		return headInfo;
	}

	unsigned int InputDim() const { return _head.InputDim(); }
	unsigned int OutputDim() const { return _tail.OutputDim(); }
	unsigned int ParamDim() const { return _head.ParamDim() + _tail.ParamDim(); }

	void SetParams( const ParamType& params )
	{
		_head.SetParams( params.headParams );
		_tail.SetParams( params.tailParams );
	}

	void SetParamsVec( const VectorType& params )
	{
		_head.SetParams( params.topRows( _head.ParamDim() ) );
		_tail.SetParams( params.bottomRows( _tail.ParamDim() ) );
	}

	ParamType GetParams() const
	{
		ParamType params;
		params.headParams = _head.GetParams();
		params.tailParams = _tail.GetParams();
		return params;
	}

	VectorType GetParamsVec() const
	{
		VectorType vec( ParamDim() );
		vec << _head.GetParamsVec(), _tail.GetParamsVec();
		return vec;
	}

	void StepParams( const VectorType& step )
	{
		_head.StepParams( step.topRows( _head.ParamDim() ) );
		_tail.StepParams( step.bottomRows( _tail.ParamDim() ) );
	}

private:

	HeadType& _head;
	TailType& _tail;

};

}