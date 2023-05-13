package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/datacollector"

	"gonum.org/v1/gonum/mat"
)

// Define the signature of a learning method.
//
// A learning method takes a set of states to learn, and returns a weight matrix update
// as well as a set of LearnStateData. Note the LearnStateData is null if
// the network is set to not collect intensive data
type LearningMethod func(*HopfieldNetwork, []*mat.VecDense) []*datacollector.LearnStateData

// Define the different learning method options.
//
// This method allows us to specify a learning method via the command line, and the Hopfield Builder.
type LearningMethodEnum int

const (
	FullSetMethod        LearningMethodEnum = iota
	IterativeBatchMethod LearningMethodEnum = iota
)

