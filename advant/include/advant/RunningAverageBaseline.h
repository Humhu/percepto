#pragma once

#include "valu/ValuInterfaces.h"
#include "percepto_msgs/RewardStamped.h"

#include <map>

namespace percepto
{

// TODO Update
class RunningAverageBaseline
: public Critic
{
public:

	RunningAverageBaseline();

	// TODO
	RunningAverageBaseline( PolicyCritic::Ptr source );

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

	virtual double GetCritique( const ros::Time& time ) const;

private:

	double _acc;
	double _gamma;
	double _cacheTime;

	Critic::Ptr _source;
	ros::Duration _pollOffset;
	ros::Timer _pollTimer;

	bool _accInitialized;
	bool _delayStarted;
	ros::Time _startDelayTime;
	ros::Duration _initDelay;

	bool _burnInStarted;
	ros::Time _startBurnTime;
	ros::Time _lastBurnTime;
	ros::Duration _burnInDuration;

	ros::Publisher _outputPub;
	typedef std::map<ros::Time, double> CacheType;
	CacheType _cache;
	ros::Subscriber _valueSub;

	void Update( const ros::Time& time, double reward );
	void RewardCallback( const percepto_msgs::RewardStamped::ConstPtr& msg );
	double GetSpan() const;
	void TimerCallback( const ros::TimerEvent& event );

	bool IsBurnedIn() const;
};

}