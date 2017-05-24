#include <ros/ros.h>
#include <rosbag/player.h>

#include <boost/foreach.hpp>

#include "infitu/StartEvaluation.h"
#include "fieldtrack/ResetFilter.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

class BagReplayer
{
public:

	BagReplayer( ros::NodeHandle& nh, ros::NodeHandle& ph )
	{
		std::string resetService;
		_resetFilter = false;
		if( GetParam( ph, "reset_filter_service", resetService ) )
		{
			_resetFilter = true;
			_resetProxy = nh.serviceClient<fieldtrack::ResetFilter>( resetService );
			// _resetProxy.waitForExistence();
		}

		rosbag::PlayerOptions opts;
		GetParam<std::string>( ph, "prefix", opts.prefix, "" );
		GetParam( ph, "quiet", opts.quiet );
		GetParam( ph, "immediate", opts.at_once );
		GetParam( ph, "pause", opts.start_paused );
		GetParam( ph, "queue", opts.queue_size, 100 );
		GetParam( ph, "hz", opts.bag_time_frequency, 100.0 );
		GetParam( ph, "clock", opts.bag_time );
		double dur;
		GetParam( ph, "delay", dur, 0.2 );
		opts.advertise_sleep = ros::WallDuration( dur );
		GetParam( ph, "rate", opts.time_scale, 1.0 );
		GetParam<float>( ph, "start", opts.time, 0.0 );
		opts.has_time = HasParam( ph, "start" );
		opts.has_duration = GetParam( ph, "duration", opts.duration );
		GetParam( ph, "skip_empty", dur );
		opts.skip_empty = ros::Duration( dur );
		GetParam( ph, "loop", opts.loop );
		GetParam( ph, "keep_alive", opts.keep_alive );
		std::vector<std::string> topics;
		if( GetParam( ph, "topics", topics ) )
		{
			BOOST_FOREACH( const std::string & top, topics )
			{
				opts.topics.push_back( top );
			}
		}
		if( GetParam( ph, "pause_topics", topics ) )
		{
			BOOST_FOREACH( const std::string & top, topics )
			{
				opts.pause_topics.push_back( top );
			}
		}
		GetParamRequired( ph, "bags", topics );
		BOOST_FOREACH( const std::string & top, topics )
		{
			opts.bags.push_back( top );
		}

		_player = std::make_shared<rosbag::Player>( opts );

		_startServer = ph.advertiseService( "start_playback", &BagReplayer::StartCallback, this );
	}

	bool StartCallback( infitu::StartEvaluation::Request& req,
	                    infitu::StartEvaluation::Response& res )
	{
		if( _resetFilter )
		{
			fieldtrack::ResetFilter freq;
			freq.request.time_to_wait = 0;
			freq.request.filter_time = ros::Time( 0 );
			if( !_resetProxy.call( freq ) )
			{
				ROS_ERROR_STREAM( "Could not reset filter" );
				return false;
			}
		}

		try
		{
			_player->publish();
			_player->cleanup();
		}
		catch( std::runtime_error& e )
		{
			ROS_ERROR_STREAM( "Could not playback: " << e.what() );
			return false;
		}
		return true;
	}

private:

	std::shared_ptr<rosbag::Player> _player;
	ros::ServiceServer _startServer;

	bool _resetFilter;
	ros::ServiceClient _resetProxy;
};

int main( int argc, char ** argv )
{
	ros::init( argc, argv, "bag_replayer" );
	ros::NodeHandle nh, ph( "~" );
	BagReplayer br( nh, ph );
	ros::spin();
}