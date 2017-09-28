#include <ros/ros.h>

#include "broadcast/BroadcastMultiReceiver.h"
#include "valu/RewardInterpolater.h"
#include "valu/ApproximateValueManager.h"
#include "advant/TDErrorCritic.h"
#include "advant/GeneralizedAdvantageCritic.h"

#include "percepto_msgs/GetCritique.h"
#include "valu/ValuInterfaces.h"

using namespace percepto;
using namespace argus;

class CriticNode
{
public:

	CriticNode( ros::NodeHandle& nh, ros::NodeHandle& ph )
	{
		ros::NodeHandle ih( ph.resolveName( "input_streams" ) );
		_receiver = std::make_shared<BroadcastMultiReceiver>();
		_receiver->Initialize( ih );

		ros::NodeHandle rh( ph.resolveName( "reward_function" ) );
		_rewards = std::make_shared<RewardInterpolater>();
		_rewards->Initialize( nh, rh );

		ros::NodeHandle vh( ph.resolveName( "value_function" ) );
		_value = std::make_shared<ApproximateValueManager>();
		_value->Initialize( _receiver, vh );
		
		ros::NodeHandle th( ph.resolveName( "td_error" ) );
		_tde = std::make_shared<TDErrorCritic>();
		_tde->Initialize( _rewards, _value, th );

		ros::NodeHandle ah( ph.resolveName( "advantage_estimator" ) );
		_critic.Initialize( _tde, ah );

		_valueServer = ph.advertiseService( "get_value",
		                                    &CriticNode::ValueCallback,
		                                    this );
		_tdServer = ph.advertiseService( "get_td_error",
		                                 &CriticNode::TDCallback,
		                                 this );
		_advantageServer = ph.advertiseService( "get_advantage", 
		                                        &CriticNode::AdvantageCallback, 
		                                        this );
	}

private:

	BroadcastMultiReceiver::Ptr _receiver;
	RewardInterpolater::Ptr _rewards;
	ApproximateValueManager::Ptr _value;
	TDErrorCritic::Ptr _tde;
	GeneralizedAdvantageCritic _critic;

	ros::ServiceServer _valueServer;
	ros::ServiceServer _tdServer;
	ros::ServiceServer _advantageServer;

	bool ValueCallback( percepto_msgs::GetCritique::Request& req,
	                    percepto_msgs::GetCritique::Response& res )
	{
		try
		{
			res.critique = _value->GetCritique( req.time );
		}
		catch( std::out_of_range e )
		{
			ROS_WARN_STREAM( "Could not get value for time: " << req.time  
			                 << std::endl << e.what() );
			return false;
		}
		return true;
	}

	bool TDCallback( percepto_msgs::GetCritique::Request& req,
	                 percepto_msgs::GetCritique::Response& res )
	{
		try
		{
			res.critique = _tde->GetCritique( req.time );
		}
		catch( std::out_of_range e )
		{
			ROS_WARN_STREAM( "Could not get TD error for time: " << req.time 
			                 << std::endl << e.what() );
			return false;
		}
		return true;
	}

	bool AdvantageCallback( percepto_msgs::GetCritique::Request& req,
	                        percepto_msgs::GetCritique::Response& res )
	{
		try
		{
			res.critique = _critic.GetCritique( req.time );
		}
		catch( std::out_of_range e )
		{
			ROS_WARN_STREAM( "Could not get advantage for time: " << req.time 
			                 << std::endl << e.what() );
			return false;
		}
		return true;
	}

};

int main( int argc, char** argv )
{
	ros::init( argc, argv, "critic_node" );

	ros::NodeHandle nh, ph("~");
	CriticNode cn( nh, ph );
	ros::spin();

	return 0;
}