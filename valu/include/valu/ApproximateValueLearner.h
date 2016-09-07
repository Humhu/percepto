#pragma once

#include "valu/ValuCommon.h"
#include "valu/ApproximateValue.h"
#include "valu/ValueInfoManager.h"
#include "valu/ValueResidualModules.h"

#include "percepto_msgs/SRSTuple.h"

#include "modprop/optim/OptimizerTypes.h"
#include "modprop/optim/ParameterL2Cost.hpp"
#include "modprop/optim/StochasticMeanCost.hpp"
#include "modprop/compo/AdditiveWrapper.hpp"

namespace percepto
{

struct ApproximateValueProblem
{
	std::deque<BellmanResidualModule> modules;
	std::deque<percepto::SquaredLoss<double>> penalties;
	std::deque<percepto::AdditiveWrapper<double>> modSums;

	percepto::StochasticMeanCost<double> loss;
	percepto::ParameterL2Cost regularizer;
	percepto::AdditiveWrapper<double> objective;

	double penaltyScale;

	ApproximateValueProblem();

	void Initialize( percepto::Parameters::Ptr params,
	                 double l2Weight,
	                 unsigned int sampleSize,
	                 double penaltyWeight );

	template <class ... Args>
	void EmplaceModule( Args&&... args )
	{
		modules.emplace_back( args... );

		penalties.emplace_back();
		penalties.back().SetSource( &modules.back().estValue->GetOutputSource() );
		penalties.back().SetTarget( 0.0 );
		penalties.back().SetScale( penaltyScale );

		modSums.emplace_back();
		modSums.back().SetSourceA( &modules.back().GetOutputSource() );
		modSums.back().SetSourceB( &penalties.back() );

		loss.AddSource( &modSums.back() );
	}

	void RemoveOldest();
	size_t NumModules() const;

	void Invalidate();
	void Foreprop();
	void Backprop();
	void BackpropNatural();

	double GetOutput() const;

};

class ApproximateValueLearner
{
public:

	ApproximateValueLearner();

	void Initialize( ros::NodeHandle& nh, ros::NodeHandle& ph );

private:

	std::string _valueName;
	argus::LookupInterface _lookup;
	ValueInfoManager _infoManager;

	ApproximateValueProblem _problem;

	std::shared_ptr<percepto::AdamStepper> _stepper;
	std::shared_ptr<percepto::SimpleConvergence> _convergence;
	std::shared_ptr<percepto::AdamOptimizer> _optimizer;
	// std::shared_ptr<percepto::SimpleNaturalOptimizer> _optimizer;
	unsigned int _optimCounter;

	bool _resetStepperAfter;
	unsigned int _minModulesToOptimize;
	unsigned int _maxModulesToKeep;
	bool _clearAfterOptimize;

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
};

}