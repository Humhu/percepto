# Cython declarations for covreg regressor types

from EigenInterface cimport MatrixXd
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "covreg/LEMCRTypes.h" namespace "covreg":

    cdef cppclass LinRegd:
        LinRegd( const MatrixXd& parameters ) except +
        LinRegd( const LinRegd& other ) except +
        unsigned int InputDim()
        unsigned int OutputDim()
        unsigned int ParameterDim()
        void SetParameters( const MatrixXd& parameters ) except +
        MatrixXd GetParameters()
        MatrixXd Evaluate( const MatrixXd& features ) except +

    # This is a hack to workaround static members with c-names directly
    cdef MatrixXd LinRegdDefaults "covreg::LinRegd::create_default_parameters"( unsigned int inputDim, 
                                                                                unsigned int outputDim )

    cdef cppclass ExpLinRegd:
        ExpLinRegd( const LinRegd& baseRegressor ) except +
        ExpLinRegd( const MatrixXd& parameters ) except +
        ExpLinRegd( const ExpLinRegd& other ) except +
        unsigned int InputDim()
        unsigned int OutputDim()
        unsigned int ParameterDim()
        void SetParameters( const MatrixXd& parameters ) except +
        MatrixXd GetParameters()
        MatrixXd Evaluate( const MatrixXd& features ) except +

    # This is a hack to workaround static members with c-names directly
    cdef MatrixXd ExpLinRegdDefaults "covreg::ExpLinRegd::create_default_parameters"( unsigned int inputDim, 
                                                                                      unsigned int outputDim )

    cdef cppclass ModCholRegd:
        cppclass ParameterType:
            MatrixXd lParameters
            MatrixXd dParameters
            ParameterType() except +
            ParameterType( const ModCholRegd.ParameterType& other )
        
        cppclass InputType:
            MatrixXd lInput
            MatrixXd dInput
            InputType() except +
            InputType( const ModCholRegd.InputType& other )

        ModCholRegd( const LinRegd& lRegressor,
                      const ExpLinRegd& dRegressor,
                      const MatrixXd& offset ) except +
        ModCholRegd( const ModCholRegd.ParameterType& params,
                      const MatrixXd& offset ) except +
        ModCholRegd( const ModCholRegd& other ) except +
        unsigned int InputDim()
        unsigned int OutputDim()
        unsigned int ParameterDim()
        void SetParameters( const ModCholRegd.ParameterType& parameters ) except +
        ModCholRegd.ParameterType GetParameters()
        MatrixXd Evaluate( const ModCholRegd.InputType& features ) except +

    cdef ModCholRegd.ParameterType ModCholRegdDefaults "covreg::ModCholRegd::create_default_parameters"( unsigned int lInputDim,
                                                                                                         unsigned int dInputDim,
                                                                                                         unsigned int outputDim )

    cdef cppclass DampedModCholRegd:
        cppclass InputType:
            ModCholRegd.InputType input
            MatrixXd offset

    cdef cppclass TransModCholRegd:
        cppclass InputType:
            ModCholRegd.InputType input
            MatrixXd transform

    cdef cppclass AffineModCholRegd:
        cppclass InputType:
            TransModCholRegd.InputType input
            MatrixXd offset

    cdef cppclass SummedModCholRegd:
        cppclass InputTuple:
            TransModCholRegd.InputType input
            bool valid

    cdef cppclass ChainedModCholRegd:
        cppclass InputType:
            vector[SummedModCholRegd.InputTuple] input
            MatrixXd offset
