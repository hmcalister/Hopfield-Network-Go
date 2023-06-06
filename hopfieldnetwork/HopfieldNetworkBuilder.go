package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/domain"
	"hmcalister/hopfield/hopfieldnetwork/noiseapplication"
	"hmcalister/hopfield/hopfieldnetwork/states/statemanager"
	"log"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

type HopfieldNetworkBuilder struct {
	randMatrixInit                 bool
	dimension                      int
	domain                         domain.DomainEnum
	forceSymmetric                 bool
	forceZeroDiagonal              bool
	learningMethod                 LearningMethod
	learningRule                   LearningRule
	epochs                         int
	maximumRelaxationUnstableUnits int
	maximumRelaxationIterations    int
	learningRate                   float64
	learningNoiseMethod            noiseapplication.NoiseApplicationMethod
	learningNoiseScale             float64
	unitsUpdatedPerStep            int
	dataCollector                  *datacollector.DataCollector
	logger                         *log.Logger
	allowIntensiveDataCollection   bool
}

// Get a new HopfieldNetworkBuilder filled with the default values.
//
// Note that some default values will cause build errors - this is intentional!
// Users should explicitly set at least these values before building.
func NewHopfieldNetworkBuilder() *HopfieldNetworkBuilder {
	return &HopfieldNetworkBuilder{
		randMatrixInit:                 false,
		dimension:                      0,
		domain:                         domain.BipolarDomain,
		forceSymmetric:                 true,
		forceZeroDiagonal:              true,
		maximumRelaxationUnstableUnits: 0,
		maximumRelaxationIterations:    100,
		learningRate:                   1.0,
		unitsUpdatedPerStep:            1,
		dataCollector:                  datacollector.NewDataCollector(),
		logger:                         log.Default(),
		allowIntensiveDataCollection:   false,
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

// Set domain of the network.
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkDomain(domain domain.DomainEnum) *HopfieldNetworkBuilder {
	networkBuilder.domain = domain
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

// Set the learning method of this network based on the LearningMethodEnum selected.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be specified before Build can be called.
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkLearningMethod(learningMethod LearningMethodEnum) *HopfieldNetworkBuilder {
	networkBuilder.learningMethod = getLearningMethod(learningMethod)
	return networkBuilder
}

// Set the learning rule of this network based on the LearningRuleEnum selected.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be specified before Build can be called.
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkLearningRule(learningRule LearningRuleEnum) *HopfieldNetworkBuilder {
	networkBuilder.learningRule = getLearningRule(learningRule)
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

// Set the learning rate of the network. Should be greater than 0.0
//
// Defaults to 1.0. This is typically a good enough value.
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetLearningRate(learningRate float64) *HopfieldNetworkBuilder {
	networkBuilder.learningRate = learningRate
	return networkBuilder
}

// Set the learning noise method for the network. Method is determined by the enum selected. See the function `noiseapplication.GetNoiseApplicationMethod` for details
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetLearningNoiseMethod(learningNoiseMethod noiseapplication.NoiseApplicationEnum) *HopfieldNetworkBuilder {
	networkBuilder.learningNoiseMethod = noiseapplication.GetNoiseApplicationMethod(learningNoiseMethod)
	return networkBuilder
}

// Set the learning noise ratio. This is the number of elements that are inverted in each state before relaxation.
//
// Defaults to 0.0. Must be in the range [0.0, 1.0] but should be a small value (e.g. <0.25)
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetLearningNoiseRatio(learningNoiseRatio float64) *HopfieldNetworkBuilder {
	networkBuilder.learningNoiseScale = learningNoiseRatio
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

// Set the DataCollector to be used in the network.
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetDataCollector(dataCollector *datacollector.DataCollector) *HopfieldNetworkBuilder {
	networkBuilder.dataCollector = dataCollector
	return networkBuilder
}

// Set the Logger to be used in the network.
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetLogger(logger *log.Logger) *HopfieldNetworkBuilder {
	networkBuilder.logger = logger
	return networkBuilder
}

// Set the flag relating to intensive data collection.
//
// Note this method returns the builder pointer so chained calls can be used.
func (networkBuilder *HopfieldNetworkBuilder) SetAllowIntensiveDataCollection(allowIntensiveDataCollection bool) *HopfieldNetworkBuilder {
	networkBuilder.allowIntensiveDataCollection = allowIntensiveDataCollection
	return networkBuilder
}

// Build and return a new HopfieldNetwork using the parameters specified with builder methods.
func (networkBuilder *HopfieldNetworkBuilder) Build() *HopfieldNetwork {
	if networkBuilder.dimension <= 0 {
		panic("HopfieldNetworkBuilder encountered an error during build! Dimension must be explicitly set to a positive integer!")
	}

	if networkBuilder.epochs <= 0 {
		panic("HopfieldNetworkBuilder encountered an error during build! Epochs must be a positive integer!")
	}

	if networkBuilder.learningNoiseScale < 0.0 || networkBuilder.learningNoiseScale > 1.0 {
		panic("HopfieldNetworkBuilder encountered an error during build! learningNoiseRatio must be in range [0.0, 1.0]!")
	}

	if networkBuilder.unitsUpdatedPerStep < 0 || networkBuilder.unitsUpdatedPerStep > networkBuilder.dimension {
		panic("HopfieldNetworkBuilder encountered an error during build! unitsUpdatedPerStep must be a positive integer that is smaller than the network dimension!")
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

	return &HopfieldNetwork{
		matrix:                         matrix,
		dimension:                      networkBuilder.dimension,
		domain:                         networkBuilder.domain,
		domainStateManager:             statemanager.GetDomainStateManager(networkBuilder.domain),
		forceSymmetric:                 networkBuilder.forceSymmetric,
		forceZeroDiagonal:              networkBuilder.forceZeroDiagonal,
		learningMethod:                 networkBuilder.learningMethod,
		learningRule:                   networkBuilder.learningRule,
		epochs:                         networkBuilder.epochs,
		randomGenerator:                randomGenerator,
		maximumRelaxationUnstableUnits: networkBuilder.maximumRelaxationUnstableUnits,
		maximumRelaxationIterations:    networkBuilder.maximumRelaxationIterations,
		learningRate:                   networkBuilder.learningRate,
		learningNoiseMethod:            networkBuilder.learningNoiseMethod,
		learningNoiseScale:             networkBuilder.learningNoiseScale,
		unitsUpdatedPerStep:            networkBuilder.unitsUpdatedPerStep,
		dataCollector:                  networkBuilder.dataCollector,
		logger:                         networkBuilder.logger,
		allowIntensiveDataCollection:   networkBuilder.allowIntensiveDataCollection,
	}

}
