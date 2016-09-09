#pragma once

#include "modprop/ModpropTypes.h"
#include "percepto_msgs/SRSTuple.h"

#include <iostream>

namespace percepto
{

struct SRSTuple
{
	typedef percepto_msgs::SRSTuple MsgType;

	ros::Time time;
	VectorType state;
	double reward;
	ros::Time nextTime;
	VectorType nextState;

	SRSTuple();
	SRSTuple( const MsgType& msg );
	MsgType ToMsg() const;
};

std::ostream& operator<<( std::ostream& os, const SRSTuple& srs );

}