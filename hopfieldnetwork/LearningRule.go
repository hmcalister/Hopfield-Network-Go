package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/states/statemanager"

	"gonum.org/v1/gonum/mat"
)

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
	HebbianLearningRule LearningRuleEnum = iota
	DeltaLearningRule   LearningRuleEnum = iota
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
		HebbianLearningRule: hebbian,
		DeltaLearningRule:   delta,
	}

	return learningRuleMap[learningRule]
}

// Compute the Hebbian weight update.
func hebbian(network *HopfieldNetwork, states []*mat.VecDense) {

	updatedMatrix := mat.NewDense(network.dimension, network.dimension, nil)
	updatedBias := mat.NewVecDense(network.dimension, nil)
	updatedMatrix.Zero()
	updatedBias.Zero()

	for _, state := range states {
		updatedMatrix.RankOne(updatedMatrix, 1, state, state)
		updatedBias.AddVec(updatedBias, state)
	}

	updatedMatrix.Scale(network.learningRate, updatedMatrix)
	updatedBias.ScaleVec(network.learningRate, updatedBias)
	network.matrix.Add(network.matrix, updatedMatrix)
	network.bias.AddVec(network.bias, updatedBias)
	network.enforceConstraints()
}

// Compute the Delta learning rule update for a network.
func delta(network *HopfieldNetwork, states []*mat.VecDense) {
	bipolarManager := &statemanager.BipolarStateManager{}

	// Create and zero out a new matrix to use as the updated weight matrix (after training)
	updatedMatrix := mat.NewDense(network.dimension, network.dimension, nil)
	updatedMatrix.Zero()

	updatedBias := mat.NewVecDense(network.dimension, nil)
	updatedBias.Zero()

	// Create a couple of vectors for use in relaxing states
	relaxationDifference := mat.NewVecDense(network.dimension, nil)

	// Make a copy of each target state so we can relax these without affecting the originals
	relaxedStates := make([]*mat.VecDense, len(states))
	for stateIndex := range relaxedStates {
		relaxedStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
		// We also apply some noise to the state to aide in learning
		network.learningNoiseMethod(network.randomGenerator, relaxedStates[stateIndex], network.learningNoiseScale)
		network.domainStateManager.ActivationFunction(relaxedStates[stateIndex])
	}

	for stateIndex := range states {
		state := states[stateIndex]

		a := mat.VecDenseCopyOf(state)
		b := mat.VecDenseCopyOf(state)
		network.UpdateState(b)
		bipolarManager.ActivationFunction(a)
		bipolarManager.ActivationFunction(b)

		relaxationDifference.Zero()
		relaxationDifference.SubVec(a, b)

		for i := 0; i < network.GetDimension(); i++ {
			for j := 0; j < network.GetDimension(); j++ {
				updatedMatrix.Set(i, j, updatedMatrix.At(i, j)+relaxationDifference.AtVec(i)*a.AtVec(j))
			}
			updatedBias.SetVec(i, updatedBias.AtVec(i)+relaxationDifference.AtVec(i))
		}
	}

	updatedMatrix.Scale(network.learningRate, updatedMatrix)
	updatedBias.ScaleVec(network.learningRate, updatedBias)
	network.matrix.Add(network.matrix, updatedMatrix)
	network.bias.AddVec(network.bias, updatedBias)
	network.enforceConstraints()
}
