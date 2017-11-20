#pragma once

#include "valu/ValuCommon.h"
#include "valu/ApproximateValue.h"
#include "valu/ValueInfoManager.h"
#include "valu/ValueLearningProblem.h"

#include "optim/ModularOptimizer.h"

#include "percepto_msgs/SRSTuple.h"

#include "argus_utils/synchronization/SynchronizationTypes.h"

namespace percepto
{

class ApproximateValueLearner
{
public:

	ApproximateValueLearner();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

private:

	argus::Mutex _mutex;

	std::string _valueName;
	argus::LookupInterface _lookup;
	ValueInfoManager _infoManager;

	ApproximateValueProblem _problem;

	ModularOptimizer::Ptr _optimizer;

	unsigned int _optimCounter;

	bool _resetStepperAfter;
	unsigned int _minModulesToOptimize;
	unsigned int _maxModulesToKeep;
	bool _clearAfterOptimize;

	bool _networkInitialized;
	ApproximateValue _value;
	
	ros::Subscriber _srsSub;
	std::deque<SRSTuple> _srsBuffer;

	ros::ServiceClient _setParamsClient;

	ros::Timer _updateTimer;
	double _discountRate;

	void SRSCallback( const percepto_msgs::SRSTuple::ConstPtr& msg );

	void UpdateCallback( const ros::TimerEvent& event );
	void SampleRange( const ros::Time& start, const ros::Time& end );
	void AddSample( const ros::Time& time );
	void RunOptimization();

	void InitializeNetwork();
};

}