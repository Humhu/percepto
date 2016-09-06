#pragma once

#include <modprop/compo/LinearRegressor.hpp>
#include <modprop/compo/ConstantRegressor.hpp>
#include <modprop/compo/ExponentialWrapper.hpp>
#include <modprop/compo/OffsetWrapper.hpp>
#include <modprop/compo/ModifiedCholeskyWrapper.hpp>
#include <modprop/neural/NetworkTypes.h>

#include "argus_utils/utils/LinalgTypes.h"


namespace percepto
{

class ContinuousPolicyModule
{
public:

	typedef std::shared_ptr<ContinuousPolicyModule> Ptr;

	typedef percepto::Source<VectorType> VectorSourceType;
	typedef percepto::Source<MatrixType> MatrixSourceType;

	ContinuousPolicyModule();
	virtual ~ContinuousPolicyModule();

	virtual void SetInputSource( VectorSourceType* src ) = 0;
	virtual VectorSourceType& GetMeanSource() = 0;
	virtual MatrixSourceType& GetInfoSource() = 0;

	virtual void Foreprop() = 0;
	virtual void Invalidate() = 0;

	virtual percepto::Parameters::Ptr CreateParameters() = 0;

	virtual void InitializeMean( const VectorType& u ) = 0;
	virtual void InitializeInformation( const MatrixType& n ) = 0;

	virtual std::string Print() const = 0;

};

std::ostream& operator<<( std::ostream& os, const ContinuousPolicyModule& m );

class ConstantGaussian
: public ContinuousPolicyModule
{
public:

	typedef std::shared_ptr<ConstantGaussian> Ptr;

	percepto::ConstantVectorRegressor mean;
	percepto::ConstantVectorRegressor correlations;
	percepto::ConstantVectorRegressor logVariances;

	percepto::ExponentialWrapper variances;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> information;

	bool useCorrelations;
	percepto::Parameters::Ptr corrParams;

	ConstantGaussian( unsigned int matDim, bool useCorr = true );
	ConstantGaussian( const ConstantGaussian& other );

	virtual void SetInputSource( VectorSourceType* src );
	virtual VectorSourceType& GetMeanSource();
	virtual MatrixSourceType& GetInfoSource();

	virtual void Foreprop();
	virtual void Invalidate();

	virtual percepto::Parameters::Ptr CreateParameters();

	virtual void InitializeMean( const VectorType& u );
	virtual void InitializeInformation( const MatrixType& n );

	virtual std::string Print() const;
};

class LinearGaussian
: public ContinuousPolicyModule
{
public:

	typedef std::shared_ptr<LinearGaussian> Ptr;

	percepto::LinearRegressor mean;
	percepto::ConstantVectorRegressor correlations;
	percepto::ConstantVectorRegressor logVariances;

	percepto::ExponentialWrapper variances;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> information;

	bool useCorrelations;
	percepto::Parameters::Ptr corrParams;

	LinearGaussian( unsigned int inputDim,
	                unsigned int matDim,
	                bool useCorr = true );

	LinearGaussian( const LinearGaussian& other );

	virtual void SetInputSource( VectorSourceType* src );
	virtual VectorSourceType& GetMeanSource();
	virtual MatrixSourceType& GetInfoSource();

	virtual void Foreprop();
	virtual void Invalidate();

	virtual percepto::Parameters::Ptr CreateParameters();

	virtual void InitializeMean( const VectorType& u );
	virtual void InitializeInformation( const MatrixType& n );

	virtual std::string Print() const;
};

class FixedVarianceGaussian
: public ContinuousPolicyModule
{
public:

	typedef std::shared_ptr<FixedVarianceGaussian> Ptr;

	percepto::PerceptronNet mean;
	percepto::ConstantVectorRegressor correlations;
	percepto::ConstantVectorRegressor logVariances;

	percepto::ExponentialWrapper variances;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> information;

	bool useCorrelations;
	percepto::Parameters::Ptr corrParams;

	FixedVarianceGaussian( unsigned int inputDim,
	                       unsigned int matDim,
	                       unsigned int numHiddenLayers,
	                       unsigned int layerWidth,
	                       bool useCorr = true );

	FixedVarianceGaussian( const FixedVarianceGaussian& other );

	virtual void SetInputSource( VectorSourceType* src );
	virtual VectorSourceType& GetMeanSource();
	virtual MatrixSourceType& GetInfoSource();

	virtual void Foreprop();
	virtual void Invalidate();

	virtual percepto::Parameters::Ptr CreateParameters();

	virtual void InitializeMean( const VectorType& u );
	virtual void InitializeInformation( const MatrixType& n );

	virtual std::string Print() const;
};

class VariableVarianceGaussian
: public ContinuousPolicyModule
{
public:

	typedef percepto::Source<VectorType> VectorSourceType;
	typedef percepto::Source<MatrixType> MatrixSourceType;

	typedef std::shared_ptr<VariableVarianceGaussian> Ptr;

	percepto::PerceptronNet mean;
	percepto::PerceptronNet correlations;
	percepto::PerceptronNet logVariances;

	percepto::ExponentialWrapper variances;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> information;

	bool useCorrelations;
	percepto::Parameters::Ptr corrParams;

	VariableVarianceGaussian( unsigned int inputDim,
	                          unsigned int matDim,
	                          unsigned int numHiddenLayers,
	                          unsigned int layerWidth,
	                          bool useCorr = true );

	VariableVarianceGaussian( const VariableVarianceGaussian& other );

	virtual void SetInputSource( VectorSourceType* src );
	virtual VectorSourceType& GetMeanSource();
	virtual MatrixSourceType& GetInfoSource();

	virtual void Foreprop();
	virtual void Invalidate();

	virtual percepto::Parameters::Ptr CreateParameters();

	virtual void InitializeMean( const VectorType& u );
	virtual void InitializeInformation( const MatrixType& n );

	virtual std::string Print() const;
};

}