#include "poli/PoliCommon.h"
#include "argus_utils/utils/MatrixUtils.h"

using namespace argus;

namespace percepto
{

ParamAction::ParamAction() {}

ParamAction::ParamAction( const ros::Time& t, const VectorType& in )
: time( t ), input( in ) {}

ContinuousAction::ContinuousAction() {}

ContinuousAction::ContinuousAction( const MsgType& msg )
: ParamAction( msg.header.stamp, GetVectorView( msg.input ) ),
  output( GetVectorView( msg.output ) ) {}

ContinuousAction::ContinuousAction( const ros::Time& t,
                                              const VectorType& in,
                                              const VectorType& out )
: ParamAction( t, in ), output( out ) {}

ContinuousAction::MsgType ContinuousAction::ToMsg() const
{
	MsgType msg;
	msg.header.stamp = time;
	SerializeMatrix( input, msg.input );
	SerializeMatrix( output, msg.output );
	return msg;
}

void ContinuousAction::Normalize( const VectorType& scales,
                                  const VectorType& offsets )
{
	output = ( ( output - offsets ).array() / scales.array() ).matrix();
}

void ContinuousAction::Unnormalize( const VectorType& scales,
                                    const VectorType& offsets )
{
	output = ( output.array() * scales.array() ).matrix() + offsets;
}

DiscreteAction::DiscreteAction() {}

DiscreteAction::DiscreteAction( const MsgType& msg )
: ParamAction( msg.header.stamp, GetVectorView( msg.input ) ),
  index( msg.index ) {}

DiscreteAction::DiscreteAction( const ros::Time& t,
                                          const VectorType& in,
                                          unsigned int ind )
: ParamAction( t, in ), index( ind ) {}

DiscreteAction::MsgType DiscreteAction::ToMsg() const
{
	MsgType msg;
	msg.header.stamp = time;
	SerializeMatrix( input, msg.input );
	msg.index = index;
	return msg;
}

}