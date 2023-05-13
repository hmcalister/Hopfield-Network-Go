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

// Map an option from the LearningMethodEnum to the specific learning method.
func getLearningMethod(learningMethod LearningMethodEnum) LearningMethod {
	learningMethodMap := map[LearningMethodEnum]LearningMethod{
		FullSetMethod: fullSetLearningMethod,
		// IterativeBatchMethod: iterativeBatch,
	}

	return learningMethodMap[learningMethod]
}

// Full Set Learning presents the entire set of states to learn at once.
func fullSetLearningMethod(network *HopfieldNetwork, states []*mat.VecDense) []*datacollector.LearnStateData {
	learnStateData := []*datacollector.LearnStateData{}
	for epoch := 0; epoch < network.epochs; epoch++ {
		learningRuleResult := network.learningRule(network, states)

		network.matrix.Add(network.matrix, learningRuleResult)
		network.cleanMatrix()

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
