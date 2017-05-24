#include "optim/AdamSearchDirector.h"
#include "argus_utils/utils/ParamUtils.h"

namespace argus
{

AdamSearchDirector::AdamSearchDirector()
	: _beta1(0.9), _beta2(0.999), _epsilon(1E-7)
{
	Reset();
}

template<typename InfoType>
void AdamSearchDirector::InitializeFromInfo(const InfoType& info)
{
	double b1, b2, eps;
	if(GetParam(info, "beta1", b1))
	{
		SetBeta1(b1);
	}
	if(GetParam(info, "beta2", b2))
	{
		SetBeta2(b2);
	}
	if(GetParam(info, "epsilon", eps))
	{
		SetEpsilon(eps);
	}
}

void AdamSearchDirector::Initialize(const ros::NodeHandle& ph)
{
	InitializeFromInfo(ph);
}

void AdamSearchDirector::Initialize(const YAML::Node& node)
{
	InitializeFromInfo(node);
}

void AdamSearchDirector::SetBeta1(double b1)
{
	_beta1 = b1;
}

void AdamSearchDirector::SetBeta2(double b2)
{
	_beta2 = b2;
}

void AdamSearchDirector::SetEpsilon(double eps)
{
	_epsilon = eps;
}

void AdamSearchDirector::Reset()
{
	_beta1Acc = _beta1;
	_beta2Acc = _beta2;
	_m = VectorType();
}

VectorType AdamSearchDirector::ComputeSearchDirection(OptimizationProblem& problem)
{
	VectorType gradient = problem.ComputeGradient();

	// Need to initialize state
	if(_m.size() == 0)
	{
		_m = VectorType::Zero(gradient.size());
		_v = VectorType::Zero(gradient.size());
	}

	if(gradient.size() != _m.size())
	{
		throw std::runtime_error("Gradient dimension mismatch.");
	}

	_m = _beta1 * _m + (1.0 - _beta1) * gradient;
	VectorType gradientSq = (gradient.array() * gradient.array()).matrix();
	_v = _beta2 * _v + (1.0 - _beta2) * gradientSq;

	VectorType mhat = _m / (1.0 - _beta1Acc);
	VectorType vhat = _v / (1.0 - _beta2Acc);
	_beta1Acc *= _beta1;
	_beta2Acc *= _beta2;

	VectorType ret = (mhat.array() / (vhat.array().sqrt() + _epsilon)).matrix();

	if(problem.IsMinimization())
	{
		return -ret;
	}
	else
	{
		return ret;
	}
}

}