#include "valu/ValuCommon.h"
#include "argus_utils/utils/MatrixUtils.h"

using namespace argus;

namespace percepto
{

SRSTuple::SRSTuple() {}

SRSTuple::SRSTuple( const MsgType& msg )
{
	time = msg.header.stamp;
	state = GetVectorView( msg.state );
	reward = msg.reward;
	nextTime = msg.next_time;
	nextState = GetVectorView( msg.next_state );
}

SRSTuple::MsgType SRSTuple::ToMsg() const
{
	MsgType msg;
	msg.header.stamp = time;
	SerializeMatrix( state, msg.state );
	msg.reward = reward;
	msg.next_time = nextTime;
	SerializeMatrix( nextState, msg.next_state );
	return msg;
}

std::ostream& operator<<( std::ostream& os, const SRSTuple& srs )
{
	os << "time: " << srs.time << std::endl;
	os << "state: " << srs.state.transpose() << std::endl;
	os << "reward: " << srs.reward << std::endl;
	os << "next time: " << srs.nextTime << std::endl;
	os << "next state: " << srs.nextState.transpose();
	return os;
}


}