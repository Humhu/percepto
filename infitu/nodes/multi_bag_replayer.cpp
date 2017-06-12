#include <ros/ros.h>
#include <rosbag/player.h>

#include <boost/foreach.hpp>
#include <unordered_map>

#include "infitu/StartEvaluation.h"
#include "fieldtrack/ResetFilter.h"
#include "argus_utils/utils/ParamUtils.h"

using namespace argus;

class MultiBagReplayer
{
  public:
	MultiBagReplayer(ros::NodeHandle &nh, ros::NodeHandle &ph)
	{
		std::string resetService;
		_resetFilter = false;
		if (GetParam(ph, "reset_filter_service", resetService))
		{
			_resetFilter = true;
			_resetProxy = nh.serviceClient<fieldtrack::ResetFilter>(resetService);
			// NOTE with sim time this hangs forever!
			// _resetProxy.waitForExistence();
		}
	}

	void Initialize(const ros::NodeHandle &ph)
	{
		YAML::Node infos;
		GetParam(ph, "bags", infos);
		YAML::Node::const_iterator iter;
		for (iter = infos.begin(); iter != infos.end(); ++iter)
		{
			const std::string &name = iter->first.as<std::string>();
			const YAML::Node& info = iter->second;
			rosbag::PlayerOptions opts = ParseOptions(info);

			std::shared_ptr<rosbag::Player> player;
			player = std::make_shared<rosbag::Player>(opts);
			_players.push_back(player);

			ros::ServiceServer playerServer;
			ros::NodeHandle bh(ph.resolveName(name));
			playerServer = bh.advertiseService<infitu::StartEvaluation::Request,
											   infitu::StartEvaluation::Response>("start_playback",
																				  boost::bind(&MultiBagReplayer::StartCallback, this, boost::ref(*player), _1, _2));
			_startServers.push_back(playerServer);
		}
	}

	rosbag::PlayerOptions ParseOptions(const YAML::Node& ph )
	{
		rosbag::PlayerOptions opts;
		GetParam<std::string>(ph, "prefix", opts.prefix, "");
		GetParam(ph, "quiet", opts.quiet, false);
		GetParam(ph, "immediate", opts.at_once, false);
		GetParam(ph, "pause", opts.start_paused, false);
		GetParam(ph, "queue", opts.queue_size, 100);
		GetParam(ph, "hz", opts.bag_time_frequency, 100.0);
		GetParam(ph, "clock", opts.bag_time, false);
		double dur;
		GetParam(ph, "delay", dur, 0.2);
		opts.advertise_sleep = ros::WallDuration(dur);
		GetParam(ph, "rate", opts.time_scale, 1.0);
	
		GetParam<float>(ph, "start", opts.time, 0.0);
		opts.has_time = HasParam(ph, "start");
	
		opts.has_duration = GetParam(ph, "duration", opts.duration);

		if( GetParam(ph, "skip_empty", dur) )
		{
			opts.skip_empty = ros::Duration(dur);
		}
		GetParam(ph, "loop", opts.loop, false);
		GetParam(ph, "keep_alive", opts.keep_alive, false);
		
		std::vector<std::string> topics;
		if (GetParam(ph, "topics", topics))
		{
			BOOST_FOREACH (const std::string &top, topics)
			{
				opts.topics.push_back(top);
			}
		}
		if (GetParam(ph, "pause_topics", topics))
		{
			BOOST_FOREACH (const std::string &top, topics)
			{
				opts.pause_topics.push_back(top);
			}
		}
		GetParamRequired(ph, "bags", topics);
		BOOST_FOREACH (const std::string &top, topics)
		{
			opts.bags.push_back(top);
		}
		return opts;
	}

	bool StartCallback(rosbag::Player &player,
					   infitu::StartEvaluation::Request &req,
					   infitu::StartEvaluation::Response &res)
	{
		if (_resetFilter)
		{
			fieldtrack::ResetFilter freq;
			freq.request.time_to_wait = 0;
			freq.request.filter_time = ros::Time(0);
			if (!_resetProxy.call(freq))
			{
				ROS_ERROR_STREAM("Could not reset filter");
				return false;
			}
		}

		try
		{
			player.publish();
			player.cleanup();
		}
		catch (std::runtime_error &e)
		{
			ROS_ERROR_STREAM("Could not playback: " << e.what());
			return false;
		}
		return true;
	}

  private:
	std::vector<std::shared_ptr<rosbag::Player>> _players;
	std::vector<ros::ServiceServer> _startServers;

	bool _resetFilter;
	ros::ServiceClient _resetProxy;
};

int main(int argc, char **argv)
{
	ros::init(argc, argv, "multi_bag_replayer");
	ros::NodeHandle nh, ph("~");
	MultiBagReplayer br(nh, ph);
	br.Initialize(ph);
	ros::spin();
}