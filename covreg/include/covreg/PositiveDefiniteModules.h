#pragma once

#include <modprop/compo/LinearRegressor.hpp>
#include <modprop/compo/ConstantRegressor.hpp>
#include <modprop/compo/ExponentialWrapper.hpp>
#include <modprop/compo/HingeExponentialWrapper.hpp>
#include <modprop/compo/OffsetWrapper.hpp>
#include <modprop/compo/ModifiedCholeskyWrapper.hpp>
#include <modprop/neural/NetworkTypes.h>

#include "argus_utils/utils/LinalgTypes.h"

namespace percepto
{

class PosDefModule
: public percepto::Source<MatrixType>
{
public:

	typedef percepto::Source<VectorType> InputSourceType;
	typedef percepto::Source<MatrixType> OutputSourceType;
	typedef percepto::Sink<MatrixType> SinkType;

	PosDefModule();
	virtual ~PosDefModule();

	virtual percepto::ParameterWrapper::Ptr CreateParameters() = 0;

	virtual void SetSource( InputSourceType* s ) = 0;

	virtual void Foreprop();

	virtual void BackpropImplementation( const MatrixType& nextDodx );

protected:

	SinkType _outputPort;

	void SetOutputModule( OutputSourceType* outputModule );

private:

	PosDefModule( const PosDefModule& other );
	PosDefModule& operator=( const PosDefModule& other );

};

class ConstantPosDefModule
: public PosDefModule
{
public:

	typedef std::shared_ptr<ConstantPosDefModule> Ptr;

	percepto::ConstantVectorRegressor lReg;
	percepto::ConstantVectorRegressor dReg;
	// percepto::HingeExponentialWrapper<VectorType> expModule;
	percepto::ExponentialWrapper expModule;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> pdModule;

	// Need to:
	// dInput.SetOutput
	// lReg.SetParameters
	// dReg.SetParameters
	ConstantPosDefModule( unsigned int matDim );

	// Copy assignment rewires all connections and should result in
	// shared parameters with the original
	ConstantPosDefModule( const ConstantPosDefModule& other );

	virtual percepto::ParameterWrapper::Ptr CreateParameters();

	virtual void SetSource( InputSourceType* s );

	virtual void Invalidate();

	// Overridden to handle the fake input port trigger
	virtual void Foreprop();

private:

	percepto::Sink<VectorType> _inputPort;
};

class LinearPosDefModule
: public PosDefModule
{
public:

	typedef percepto::Source<VectorType> InputSourceType;
	typedef percepto::Source<MatrixType> OutputSourceType;
	typedef percepto::Sink<MatrixType> SinkType;

	typedef std::shared_ptr<LinearPosDefModule> Ptr;

	percepto::TerminalSource<VectorType> dInput;
	percepto::ConstantVectorRegressor lReg;
	percepto::LinearRegressor dReg;
	// percepto::HingeExponentialWrapper<VectorType> expModule;
	percepto::ExponentialWrapper expModule;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> pdModule;

	LinearPosDefModule( unsigned int inputDim, 
	                     unsigned int matDim );

	// Copy assignment rewires all connections and should result in
	// shared parameters with the original
	LinearPosDefModule( const LinearPosDefModule& other );

	virtual percepto::ParameterWrapper::Ptr CreateParameters();

	virtual void SetSource( InputSourceType* s );

	virtual void Invalidate();

	virtual void Foreprop();

	virtual void BackpropImplementation( const MatrixType& nextDodx );

private:

	percepto::Sink<VectorType> _inputPort;

};

class VarReLUPosDefModule
: public PosDefModule
{
public:

	typedef percepto::Source<VectorType> InputSourceType;
	typedef percepto::Source<MatrixType> OutputSourceType;
	typedef percepto::Sink<MatrixType> SinkType;

	typedef std::shared_ptr<VarReLUPosDefModule> Ptr;

	percepto::TerminalSource<VectorType> dInput;
	percepto::ConstantVectorRegressor lReg;
	percepto::ReLUNet dReg;
	// percepto::PerceptronNet dReg;
	// percepto::HingeExponentialWrapper<VectorType> expModule;
	percepto::ExponentialWrapper expModule;
	percepto::ModifiedCholeskyWrapper psdModule;
	percepto::OffsetWrapper<MatrixType> pdModule;

	VarReLUPosDefModule( unsigned int inputDim,
	                     unsigned int matDim,
	                     unsigned int numHiddenLayers,
	                     unsigned int layerWidth );

	// Copy assignment rewires all connections and should result in
	// shared parameters with the original
	// dReg.SetParameters
	// dReg.SetParameters
	VarReLUPosDefModule( const VarReLUPosDefModule& other );

	virtual percepto::ParameterWrapper::Ptr CreateParameters();

	virtual void SetSource( InputSourceType* s );

	virtual void Invalidate();

	virtual void Foreprop();

	virtual void BackpropImplementation( const MatrixType& nextDodx );

private:

	percepto::Sink<VectorType> _inputPort;

};

}