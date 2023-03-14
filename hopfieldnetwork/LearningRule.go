package hopfieldnetwork

import (
	"gonum.org/v1/gonum/mat"
)

const private_DELTA_THREADS = 8

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

// Map an option from the LearningRule enum to the specific learning rule
//
// # Arguments
//
// * `learningRule`: The learning rule selected
//
// # Returns
//
// The learning rule from the family specified
func getLearningRule(learningRule LearningRuleEnum) LearningRule {
	learningRuleMaps := map[LearningRuleEnum]LearningRule{
		HebbianLearningRule: hebbian,
		DeltaLearningRule:   delta,
	}

	return learningRuleMaps[learningRule]
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
func hebbian(network *HopfieldNetwork, states []*mat.VecDense) *mat.Dense {
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

	relaxationDifference := mat.NewVecDense(network.dimension, nil)
	stateContribution := mat.NewDense(network.dimension, network.dimension, nil)

	relaxedStates := make([]*mat.VecDense, len(states))
	for stateIndex := range states {
		relaxedStates[stateIndex] = mat.VecDenseCopyOf(states[stateIndex])
	}

	relaxationResults := network.ConcurrentRelaxStates(relaxedStates, private_DELTA_THREADS)

	for stateIndex := range states {
		state := states[stateIndex]
		stateHistory := relaxationResults[stateIndex].StateHistory
		relaxedState := stateHistory[len(stateHistory)-1]

		relaxationDifference.Zero()
		relaxationDifference.SubVec(state, relaxedState)

		stateContribution.Zero()
		stateContribution.Outer(0.5, relaxationDifference, state)

		updatedMatrix.Add(updatedMatrix, stateContribution)
	}

	return updatedMatrix
}
