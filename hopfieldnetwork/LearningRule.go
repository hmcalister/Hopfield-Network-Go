package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/states/statemanager"
	"math"

	"gonum.org/v1/gonum/mat"
)

const private_THERMAL_DELTA_TEMPERATURE = 1.0

// Define a learning rule as a function taking a network along with a collection of states.
//
// The network is update IN the learning method: nothing is returned!
//
// # Arguments
//
// Network *HopfieldNetwork: A Hopfield Network that will be learned. This argument is required for some learning rules.
//
// States []*mat.VecDense: A slice of states to try and learn.
type LearningRule func(*HopfieldNetwork, []*mat.VecDense)

// Define the different learning rule options.
//
// This enum allows for the user to select a learning rule via the builder interface,
// but is also used to map from type of learning rule to specific implementation
// based on network domain.
//
// The integer type is redefined here to avoid type mismatches in code.
type LearningRuleEnum int

const (
	HebbianLearningRule                   LearningRuleEnum = iota
	BipolarMappedHebbianLearningRule      LearningRuleEnum = iota
	DeltaLearningRule                     LearningRuleEnum = iota
	BipolarMappedDeltaLearningRule        LearningRuleEnum = iota
	ThermalDeltaLearningRule              LearningRuleEnum = iota
	BipolarMappedThermalDeltaLearningRule LearningRuleEnum = iota
)

// Map an option from the LearningRule enum to the specific learning rule
//
// # Arguments
//
// learningRule LearningRuleEnum: The learning rule selected
//
// # Returns
//
// The learning rule from the family specified
func getLearningRule(learningRule LearningRuleEnum) LearningRule {
	learningRuleMap := map[LearningRuleEnum]LearningRule{
		HebbianLearningRule:                   hebbian,
		BipolarMappedHebbianLearningRule:      bipolarMappedHebbian,
		DeltaLearningRule:                     delta,
		BipolarMappedDeltaLearningRule:        bipolarMappedDelta,
		ThermalDeltaLearningRule:              thermalDelta,
		BipolarMappedThermalDeltaLearningRule: bipolarMappedThermalDelta,
	}

	return learningRuleMap[learningRule]
}

// Compute the Hebbian weight update.
func hebbian(network *HopfieldNetwork, states []*mat.VecDense) {

	updatedMatrix := mat.NewDense(network.dimension, network.dimension, nil)
	updatedMatrix.Zero()

	for _, state := range states {
		updatedMatrix.RankOne(updatedMatrix, 1, state, state)
	}

	updatedMatrix.Scale(network.learningRate, updatedMatrix)
	network.matrix.Add(network.matrix, updatedMatrix)
	network.enforceConstraints()
}

// Compute the bipolar mapped hebbian weight update
func bipolarMappedHebbian(network *HopfieldNetwork, states []*mat.VecDense) {

	bipolarStateManager := statemanager.BipolarStateManager{}
	mappedTargetStates := make([]*mat.VecDense, len(states))
	for stateIndex := range states {
		mappedTargetStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
		bipolarStateManager.ActivationFunction(mappedTargetStates[stateIndex])
	}
	hebbian(network, mappedTargetStates)
}

// Compute the Delta learning rule update for a network.
func delta(network *HopfieldNetwork, states []*mat.VecDense) {

	updatedMatrix := mat.NewDense(network.dimension, network.dimension, nil)
	updatedMatrix.Zero()

	relaxationDifference := mat.NewVecDense(network.dimension, nil)

	// Make a copy of each target state so we can relax these without affecting the originals
	relaxedStates := make([]*mat.VecDense, len(states))
	for stateIndex := range relaxedStates {
		relaxedStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
		// We also apply some noise to the state to aide in learning
		network.learningNoiseMethod(network.randomGenerator, relaxedStates[stateIndex], network.learningNoiseScale)
		network.domainStateManager.ActivationFunction(relaxedStates[stateIndex])
		network.UpdateState(relaxedStates[stateIndex])
	}

	for stateIndex := range states {
		relaxationDifference.SubVec(states[stateIndex], relaxedStates[stateIndex])

		updatedMatrix.RankOne(updatedMatrix, 0.5, relaxationDifference, states[stateIndex])
	}

	updatedMatrix.Scale(network.learningRate, updatedMatrix)
	network.matrix.Add(network.matrix, updatedMatrix)
	network.enforceConstraints()
}

// Compute the bipolar mapped delta weight update
func bipolarMappedDelta(network *HopfieldNetwork, states []*mat.VecDense) {

	bipolarStateManager := statemanager.BipolarStateManager{}
	mappedTargetStates := make([]*mat.VecDense, len(states))
	for stateIndex := range states {
		mappedTargetStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
		bipolarStateManager.ActivationFunction(mappedTargetStates[stateIndex])
	}
	delta(network, mappedTargetStates)
}

// Compute the thermal Delta learning rule update for a network.
func thermalDelta(network *HopfieldNetwork, states []*mat.VecDense) {

	updatedMatrix := mat.NewDense(network.dimension, network.dimension, nil)
	updatedMatrix.Zero()

	relaxationDifference := mat.NewVecDense(network.dimension, nil)
	temperatureCalculationVector := mat.NewVecDense(network.dimension, nil)
	weightFactor := 1 / (1 + mat.Norm(network.matrix, 2.0))

	// Make a copy of each target state so we can relax these without affecting the originals
	relaxedStates := make([]*mat.VecDense, len(states))
	for stateIndex := range relaxedStates {
		relaxedStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
		// We also apply some noise to the state to aide in learning
		network.learningNoiseMethod(network.randomGenerator, relaxedStates[stateIndex], network.learningNoiseScale)
		network.domainStateManager.ActivationFunction(relaxedStates[stateIndex])
		network.UpdateState(relaxedStates[stateIndex])
	}

	for stateIndex := range states {
		relaxationDifference.SubVec(states[stateIndex], relaxedStates[stateIndex])

		temperatureCalculationVector.MulVec(network.matrix, states[stateIndex])
		temperatureFactor := math.Exp(-1.0 * weightFactor * mat.Norm(temperatureCalculationVector, 2) / (private_THERMAL_DELTA_TEMPERATURE))

		updatedMatrix.RankOne(updatedMatrix, temperatureFactor, relaxationDifference, states[stateIndex])
	}

	updatedMatrix.Scale(network.learningRate, updatedMatrix)
	network.matrix.Add(network.matrix, updatedMatrix)
	network.enforceConstraints()
}

// Compute the bipolar mapped thermal delta weight update
func bipolarMappedThermalDelta(network *HopfieldNetwork, states []*mat.VecDense) {

	bipolarStateManager := statemanager.BipolarStateManager{}
	mappedTargetStates := make([]*mat.VecDense, len(states))
	for stateIndex := range states {
		mappedTargetStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
		bipolarStateManager.ActivationFunction(mappedTargetStates[stateIndex])
	}
	thermalDelta(network, mappedTargetStates)
}
