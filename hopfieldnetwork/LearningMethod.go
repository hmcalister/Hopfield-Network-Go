package hopfieldnetwork

import (
	"hmcalister/hopfield/hopfieldnetwork/datacollector"

	"github.com/schollz/progressbar/v3"
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

// Map an option from the LearningMethodEnum to the specific learning method.
func getLearningMethod(learningMethod LearningMethodEnum) LearningMethod {
	learningMethodMap := map[LearningMethodEnum]LearningMethod{
		FullSetMethod:        fullSetLearningMethod,
		IterativeBatchMethod: iterativeBatchLearningMethod,
	}

	return learningMethodMap[learningMethod]
}

// Full Set Learning presents the entire set of states to learn at once.
func fullSetLearningMethod(network *HopfieldNetwork, states []*mat.VecDense) []*datacollector.LearnStateData {
	learnStateData := []*datacollector.LearnStateData{}
	bar := progressbar.Default(int64(network.epochs), "LEARNING EPOCHS")
	for epoch := 0; epoch < network.epochs; epoch++ {
		network.learningRule(network, states)
		bar.Add(1)

		// Learn State Data is intensive, as it involves calculating th energy at every epoch
		// Only collect if requested.
		if network.allowIntensiveDataCollection {
			tempLearnStateData := make([]*datacollector.LearnStateData, len(states))
			for stateIndex, state := range states {
				tempLearnStateData[stateIndex] = &datacollector.LearnStateData{
					Epoch:            epoch,
					TargetStateIndex: stateIndex,
					EnergyProfile:    network.AllUnitEnergies(state),
					Stable:           network.StateIsStable(state),
				}
			}
			learnStateData = append(learnStateData, tempLearnStateData...)
		}

		if network.AllStatesAreStable(states) {
			break
		}
	}
	return learnStateData
}

// Iterative batch learning divides the set of states into subsets of a certain size.
// Then, at iteration k, the first k batches are presented for a number of epochs.
// This means the first batch is presented many times, the seconds batch presented one fewer times,
// and the final batch presented only once.
//
// The parameters of this method are currently set as constants, although in future this could be
// achieved through a command line flag.
func iterativeBatchLearningMethod(network *HopfieldNetwork, states []*mat.VecDense) []*datacollector.LearnStateData {
	BATCHSIZE := 5
	NUMBATCHES := len(states) / BATCHSIZE
	learnStateData := []*datacollector.LearnStateData{}
	var statesSubset []*mat.VecDense
	totalEpochsPassed := 0

	for iteration := 1; iteration <= NUMBATCHES; iteration++ {
		statesSubset = states[:BATCHSIZE*iteration]
		// We can be sneaky here and treat this subset as a fullSet problem!
		iterationLearnStateData := fullSetLearningMethod(network, statesSubset)
		for _, data := range iterationLearnStateData {
			data.Epoch += totalEpochsPassed
		}
		totalEpochsPassed = iterationLearnStateData[len(iterationLearnStateData)-1].Epoch
		learnStateData = append(learnStateData, iterationLearnStateData...)
	}

	return learnStateData
}
