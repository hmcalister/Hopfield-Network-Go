package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"

	"gonum.org/v1/gonum/mat"
)

// Define a learning rule as a function taking a network along with a collection of states.
//
// A matrix is returned that is learned to stabilize the given states.
//
// # Arguments
//
// * `Network`: A Hopfield Network that will be learned. This argument is required for some learning rules.
// * `States`: A slice of states to try and learn.
//
// # Returns
//
// A pointer to a new matrix that stabilizes the given states as much as possible.
type LearningRule func(*HopfieldNetwork, []*mat.VecDense) *mat.Dense

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

// Map an option from the LearningRule enum to the specific learning rule based on network domain
//
// # Arguments
//
// * `learningRule`: The learning rule selected
// * `domain`: The domain of the network.
//
// # Returns
//
// The learning rule from the family specified, implemented for the selected network domain
func getLearningRule(learningRule LearningRuleEnum, domain networkdomain.NetworkDomain) LearningRule {
	domainSpecificHebbian := map[networkdomain.NetworkDomain]LearningRule{
		networkdomain.BinaryDomain:  binaryHebbian,
		networkdomain.BipolarDomain: bipolarHebbian,
	}[domain]

	learningRuleMaps := map[LearningRuleEnum]LearningRule{
		HebbianLearningRule: domainSpecificHebbian,
		DeltaLearningRule:   delta,
	}

	return learningRuleMaps[learningRule]
}

// Compute the Hebbian weight update for a binary domain network.
//
// Stores this result in the given networks weight matrix.
//
// # Arguments
//
// * `Network`: A Hopfield Network that will be learned. This argument is required for some learning rules.
// * `States`: A slice of states to try and learn.
//
// # Returns
//
// A pointer to a new matrix that stabilizes the given states as much as possible.
func binaryHebbian(network *HopfieldNetwork, states []*mat.VecDense) *mat.Dense {
	updatedMatrix := mat.DenseCopyOf(network.GetMatrix())
	updatedMatrix.Zero()
	for _, state := range states {
		for i := 0; i < network.GetDimension(); i++ {
			for j := 0; j < network.GetDimension(); j++ {
				val := (2*state.AtVec(i) - 1) * (2*state.AtVec(j) - 1)
				val += updatedMatrix.At(i, j)
				updatedMatrix.Set(i, j, val)
			}
		}
	}
	return updatedMatrix
}

// Compute the Hebbian weight update for a bipolar domain network.
//
// # Arguments
//
// * `Network`: A Hopfield Network that will be learned. This argument is required for some learning rules.
// * `States`: A slice of states to try and learn.
//
// # Returns
//
// A pointer to a new matrix that stabilizes the given states as much as possible.
func bipolarHebbian(network *HopfieldNetwork, states []*mat.VecDense) *mat.Dense {
	updatedMatrix := mat.DenseCopyOf(network.GetMatrix())
	updatedMatrix.Zero()
	for _, state := range states {
		for i := 0; i < network.GetDimension(); i++ {
			for j := 0; j < network.GetDimension(); j++ {
				val := state.AtVec(i) * state.AtVec(j)
				val += updatedMatrix.At(i, j)
				updatedMatrix.Set(i, j, val)
			}
		}
	}
	return updatedMatrix
}

// Compute the Delta learning rule update for a network.
//
// # Arguments
//
// * `Network`: A Hopfield Network that will be learned. This argument is required for some learning rules.
// * `States`: A slice of states to try and learn.
//
// # Returns
//
// A pointer to a new matrix that stabilizes the given states as much as possible.
func delta(network *HopfieldNetwork, states []*mat.VecDense) *mat.Dense {
	updatedMatrix := mat.NewDense(network.dimension, network.dimension, nil)
	updatedMatrix.Zero()

	relaxedState := mat.NewVecDense(network.dimension, nil)
	relaxationDifference := mat.NewVecDense(network.dimension, nil)
	stateContribution := mat.NewDense(network.dimension, network.dimension, nil)

	for _, state := range states {
		relaxedState.CopyVec(state)
		network.RelaxState(relaxedState)

		relaxationDifference.Zero()
		relaxationDifference.SubVec(state, relaxedState)

		stateContribution.Zero()
		stateContribution.Outer(0.5, relaxationDifference, state)

		updatedMatrix.Add(updatedMatrix, stateContribution)
	}

	return updatedMatrix
}
