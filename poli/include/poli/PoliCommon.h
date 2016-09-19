#pragma once

#include <ros/ros.h>
#include "modprop/ModpropTypes.h"
#include "percepto_msgs/ContinuousAction.h"
#include "percepto_msgs/DiscreteAction.h"

namespace percepto
{

struct ParamAction
{
	ros::Time time;
	VectorType input;

	ParamAction();
	ParamAction( const ros::Time& t,
	             const VectorType& in );
};

struct ContinuousAction
: public ParamAction
{
	typedef percepto_msgs::ContinuousAction MsgType;
	VectorType output;

	ContinuousAction();
	ContinuousAction( const ros::Time& t, 
	                       const VectorType& in,
	                       const VectorType& out );
	ContinuousAction( const MsgType& msg );

	void Normalize( const VectorType& scales,
	                const VectorType& offsets );
	void Unnormalize( const VectorType& scales,
	                 const VectorType& offsets );
	MsgType ToMsg() const;
};

struct DiscreteAction
: public ParamAction
{
	typedef percepto_msgs::DiscreteAction MsgType;

	unsigned int index;

	DiscreteAction();
	DiscreteAction( const ros::Time& t,
	                     const VectorType& in,
	                     unsigned int ind );
	DiscreteAction( const MsgType& msg );
	MsgType ToMsg() const;
};

double gaussian_kl_divergence( const VectorType& mean1, const MatrixType& info1,
                               const VectorType& mean2, const MatrixType& info2 );

}