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
type LearningRule func(HopfieldNetwork, []*mat.VecDense) *mat.Dense

