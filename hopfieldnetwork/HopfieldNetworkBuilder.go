package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/activationfunction"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type HopfieldNetworkBuilder struct {
	randMatrixInit                 bool
	dimension                      int
	forceSymmetric                 bool
	forceZeroDiagonal              bool
	domain                         networkdomain.NetworkDomain
	learningRule                   LearningRule
	epochs                         int
	maximumRelaxationUnstableUnits int
	maximumRelaxationIterations    int
	unitsUpdatedPerStep            int
}

// Get a new HopfieldNetworkBuilder filled with the default values.
//
// Note that some default values will cause build errors - this is intentional!
// Users should explicitly set at least these values before building.
func NewHopfieldNetworkBuilder() *HopfieldNetworkBuilder {
	return &HopfieldNetworkBuilder{
		randMatrixInit:                 false,
		dimension:                      0,
		forceSymmetric:                 true,
		forceZeroDiagonal:              true,
		domain:                         networkdomain.UnspecifiedDomain,
		maximumRelaxationUnstableUnits: 0,
		maximumRelaxationIterations:    100,
		unitsUpdatedPerStep:            1,
	}
}

// Set the randMatrixInit flag in the builder. If true, the new network will have a weight matrix
// initialized with random standard Gaussian values. If false (default) the matrix will have a zero weight matrix.
func (networkBuilder *HopfieldNetworkBuilder) SetRandMatrixInit(randMatrixInitFlag bool) *HopfieldNetworkBuilder {
	networkBuilder.randMatrixInit = randMatrixInitFlag
	return networkBuilder
}

// Set the dimension of the HopfieldNetwork - i.e. the dimension of the square matrix.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be set specified Build can be called
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkDimension(dimension int) *HopfieldNetworkBuilder {
	networkBuilder.dimension = dimension
	return networkBuilder
}

// Set state of the ForceSymmetric flag in the network.
//
// If true, the network will always have a symmetric weight matrix (W_ij == W_ji).
//
// This value defaults to true if not explicitly set.
func (networkBuilder *HopfieldNetworkBuilder) SetForceSymmetric(symmetricFlag bool) *HopfieldNetworkBuilder {
	networkBuilder.forceSymmetric = symmetricFlag
	return networkBuilder
}

// Set state of the ForceZeroDiagonal flag in the network.
//
// If true, the network will always have a zero-diagonal weight matrix (W_ii == 0).
//
// This value defaults to true if not explicitly set.
func (networkBuilder *HopfieldNetworkBuilder) SetForceZeroDiagonal(zeroDiagonalFlag bool) *HopfieldNetworkBuilder {
	networkBuilder.forceZeroDiagonal = zeroDiagonalFlag
	return networkBuilder
}

// Set the domain of the HopfieldNetwork - i.e. what numbers are allowed to exist in states.
//
// Valid options are taken from the NetworkDomain enum (BinaryDomain, BipolarDomain, ContinuousDomain).
// Note that UnspecifiedDomain is the default and throws and error if building is attempted.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be specified before Build can be called.
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkDomain(domain networkdomain.NetworkDomain) *HopfieldNetworkBuilder {
	networkBuilder.domain = domain
	return networkBuilder
}

// Set the learning rule of this network based on the LearningRuleEnum selected.
//
// Note this call requires the network domain to be set. If the network domain is not set, a panic occurs.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be specified before Build can be called.
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkLearningRule(learningRule LearningRuleEnum) *HopfieldNetworkBuilder {
	if networkBuilder.domain == networkdomain.UnspecifiedDomain {
		panic("HopfieldNetworkBuilder encountered an error! Domain must be explicitly set to select a learning rule!")
	}

	networkBuilder.learningRule = getLearningRule(learningRule, networkBuilder.domain)
	return networkBuilder
}

// Set the number of epochs to train for.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be specified before Build can be called.
func (networkBuilder *HopfieldNetworkBuilder) SetEpochs(epochs int) *HopfieldNetworkBuilder {
	networkBuilder.epochs = epochs
	return networkBuilder
}

// Set the maximum number of units that are allowed to be unstable for a state to be considered relaxed.
//
// Defaults to 0 (state must be perfectly stable). Typically this value should be around 0.01 - 0.1 of the network dimension
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetMaximumRelaxationUnstableUnits(maximumRelaxationUnstableUnits int) *HopfieldNetworkBuilder {
	networkBuilder.maximumRelaxationUnstableUnits = maximumRelaxationUnstableUnits
	return networkBuilder
}

// Set the maximum number iterations allowed to occur before erroring out from the relaxation.
//
// Defaults to 100. This is typically a good enough value.
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetMaximumRelaxationIterations(maximumRelaxationIterations int) *HopfieldNetworkBuilder {
	networkBuilder.maximumRelaxationIterations = maximumRelaxationIterations
	return networkBuilder
}

// Set the number of units that are update by each step / each matrix multiplication.
//
// Default to 1. This is the typical Hopfield behavior and is assured to be stable given enough time.
// Values larger than 1 may result in poor performance. Do not use a value larger than the dimension of the network.
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetUnitsUpdatedPerStep(unitsUpdatedPerStep int) *HopfieldNetworkBuilder {
	networkBuilder.unitsUpdatedPerStep = unitsUpdatedPerStep
	return networkBuilder
}

// Build and return a new HopfieldNetwork using the parameters specified with builder methods.
func (networkBuilder *HopfieldNetworkBuilder) Build() HopfieldNetwork {
	if networkBuilder.dimension <= 0 {
		panic("HopfieldNetworkBuilder encountered an error during build! Dimension must be explicitly set to a positive integer!")
	}

	if networkBuilder.domain == networkdomain.UnspecifiedDomain {
		panic("HopfieldNetworkBuilder encountered an error during build! Domain must be explicitly set to a valid network domain!")
	}

	if networkBuilder.epochs <= 0 {
		panic("HopfieldNetworkBuilder encountered an error during build! Epochs must be a positive integer!")
	}

	if networkBuilder.unitsUpdatedPerStep < 0 || networkBuilder.unitsUpdatedPerStep > networkBuilder.dimension {
		panic("HopfieldNetworkBuilder encountered an error during build! unitsUpdatedPerStep must be aa positive integer that is smaller than the network dimension!")
	}

	randSrc := rand.NewSource((uint64(time.Now().UnixNano())))
	randomGenerator := rand.New(randSrc)

	var matrix *mat.Dense
	if networkBuilder.randMatrixInit {
		normalDistribution := distuv.Normal{
			Mu:    0,
			Sigma: 0.01,
			Src:   randSrc,
		}
		matrixData := make([]float64, networkBuilder.dimension*networkBuilder.dimension)
		for i := range matrixData {
			matrixData[i] = normalDistribution.Rand()
		}
		matrix = mat.NewDense(networkBuilder.dimension, networkBuilder.dimension, matrixData)
	} else {
		matrix = mat.NewDense(networkBuilder.dimension, networkBuilder.dimension, nil)
		matrix.Zero()
	}

	activationFunction := activationfunction.DomainToActivationFunctionMap[networkBuilder.domain]

	return HopfieldNetwork{
		matrix:                         matrix,
		dimension:                      networkBuilder.dimension,
		forceSymmetric:                 networkBuilder.forceSymmetric,
		forceZeroDiagonal:              networkBuilder.forceZeroDiagonal,
		domain:                         networkBuilder.domain,
		learningRule:                   networkBuilder.learningRule,
		epochs:                         networkBuilder.epochs,
		activationFunction:             activationFunction,
		randomGenerator:                randomGenerator,
		maximumRelaxationUnstableUnits: networkBuilder.maximumRelaxationUnstableUnits,
		maximumRelaxationIterations:    networkBuilder.maximumRelaxationIterations,
		unitsUpdatedPerStep:            networkBuilder.unitsUpdatedPerStep,
	}

}
