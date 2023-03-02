// A collection of structures and functions to implement the Hopfield Network in Go
package hopfieldnetwork

import (
	"fmt"
	"hmcalister/hopfield/hopfieldnetwork/activationfunction"
	"hmcalister/hopfield/hopfieldnetwork/datacollector"
	"hmcalister/hopfield/hopfieldnetwork/energyfunction"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
	"hmcalister/hopfield/hopfieldutils"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// ------------------------------------------------------------------------------------------------
// STRUCT DEFINITION
// ------------------------------------------------------------------------------------------------

// A representation of a Hopfield Network.
//
// Should be created using the HopfieldNetworkBuilder methods.
type HopfieldNetwork struct {
	matrix                         *mat.Dense
	dimension                      int
	forceSymmetric                 bool
	forceZeroDiagonal              bool
	domain                         networkdomain.NetworkDomain
	learningRule                   LearningRule
	epochs                         int
	maximumRelaxationUnstableUnits int
	maximumRelaxationIterations    int
	unitsUpdatedPerStep            int
	activationFunction             activationfunction.ActivationFunction
	randomGenerator                *rand.Rand
	learnedStates                  []*mat.VecDense
	dataCollector                  *datacollector.DataCollector
}

// ------------------------------------------------------------------------------------------------
// INTERNAL METHODS
// ------------------------------------------------------------------------------------------------

// Fixes the networks matrix according to the forceSymmetric and forceZeroDiagonal properties set.
func (network *HopfieldNetwork) cleanMatrix() {
	if network.forceZeroDiagonal {
		for i := 0; i < network.dimension; i++ {
			network.matrix.Set(i, i, 0.0)
		}
	}

	if network.forceSymmetric {
		for i := 0; i < network.dimension-1; i++ {
			for j := i; j < network.dimension; j++ {
				network.matrix.Set(j, i, network.matrix.At(i, j))
			}
		}
	}
}

// Create an return an array of integers that contains every unit index once.
// This is useful for updating units in a random order - simply shuffle this list and iterate!
func (network *HopfieldNetwork) getUnitIndices() []int {
	unitIndices := make([]int, network.dimension)
	for i := 0; i < network.dimension; i++ {
		unitIndices[i] = i
	}
	return unitIndices
}

// ------------------------------------------------------------------------------------------------
// GETTERS
// ------------------------------------------------------------------------------------------------

// Get a reference to the weight matrix of this network.
//
// Note that this gives a reference to the matrix
// meaning the caller can update the matrix!
//
// This behavior may change in future.
//
// # Returns
//
// A references to the matrix of this network
func (network *HopfieldNetwork) GetMatrix() *mat.Dense {
	return network.matrix
}

// Get the dimension of the network
//
// # Returns
//
// The dimension of this network as an int
func (network *HopfieldNetwork) GetDimension() int {
	return network.dimension
}

// Get the learned states of this network
//
// Note that this gives a reference to the matrix
// meaning the caller can update the matrix!
//
// # Returns
//
// A list of vectors representing the learned states of this network
func (network *HopfieldNetwork) GetLearnedStates() []*mat.VecDense {
	return network.learnedStates
}

// Implement Stringer for nicer formatting
func (network *HopfieldNetwork) String() string {
	return fmt.Sprintf("Hopfield Network\n\tDimension: %d\n\tDomain: %s\n",
		network.dimension,
		network.domain.String())
}

// ------------------------------------------------------------------------------------------------
// METHODS ON STATES / STATE ENERGIES
// ------------------------------------------------------------------------------------------------

// Get the energy of a given state with respect to the network matrix.
//
// # Arguments
//
// * `state`: The vector to measure the energy of.
//
// # Returns
//
// A float64 representing the energy of the given state with respect to the network.
// Note a lower energy is more stable - but a negative state energy may still be unstable!
func (network *HopfieldNetwork) StateEnergy(state *mat.VecDense) float64 {
	return energyfunction.StateEnergy(network.matrix, state)
}

// Get the energy of a given unit (indexed by i) in the state with respect to the network matrix.
//
// # Arguments
//
// * `state`: The vector to measure the energy of.
// * `unit_index`: The unit index into the vector to measure.
//
// # Returns
//
// A float64 representing the energy of the given unit within the state.
func (network *HopfieldNetwork) UnitEnergy(state *mat.VecDense, unit_index int) float64 {
	return energyfunction.UnitEnergy(network.matrix, state, unit_index)
}

// Get the energy of a each unit within a state with respect to the network matrix.
//
// # Arguments
//
// * `state`: The vector to measure the energy of.
//
// # Returns
//
// A slice of float64 representing the energy of the given state's units with respect to the network.
func (network *HopfieldNetwork) AllUnitEnergies(state *mat.VecDense) []float64 {
	unitEnergies := energyfunction.AllUnitEnergies(network.matrix, state)
	return unitEnergies.RawVector().Data
}

// Determine if a given state is unstable.
//
// Checks the number of units with positive energy against
// the number of allowable unstable units in the network parameters
//
// # Arguments
//
// * `state`: A state to check the stability of
//
// # Returns
//
// The stability of the state, true for stable, false for unstable
func (network *HopfieldNetwork) StateIsStable(state *mat.VecDense) bool {
	stateEnergies := network.AllUnitEnergies(state)

	unstableCount := 0
	for _, energy := range stateEnergies {
		if energy > 0 {
			unstableCount += 1
		}
	}

	return unstableCount <= network.maximumRelaxationUnstableUnits
}

// Determine if ALL states in the given list are stable.
//
// # Arguments
//
// * `States`: A list of states to check
//
// # Returns
//
// True if all states in the list are stable, false if any state is unstable
func (network *HopfieldNetwork) AllStatesAreStable(states []*mat.VecDense) bool {
	for _, state := range states {
		if !network.StateIsStable(state) {
			return false
		}
	}
	return true
}

// ------------------------------------------------------------------------------------------------
// LEARNING METHODS
// ------------------------------------------------------------------------------------------------

// Update the weight matrix of the network to learn a new set of states.
//
// Note this implementation currently simply adds the learning rule output
// to the current weight matrix.
//
// # Arguments
//
// * `states`: A collection of states to learn
func (network *HopfieldNetwork) LearnStates(states []*mat.VecDense) {
	network.learnedStates = append(network.learnedStates, states...)
	for _epoch := 0; _epoch < network.epochs; _epoch++ {
		learningRuleResult := network.learningRule(network, states)
		network.matrix.Add(network.matrix, learningRuleResult)
		network.cleanMatrix()

		if network.AllStatesAreStable(states) {
			return
		}
	}
}

// ------------------------------------------------------------------------------------------------
// STATE UPDATE AND RELAXATION METHODS
// ------------------------------------------------------------------------------------------------

type RelaxationResult struct {
	Stable   bool
	NumSteps int
}

// Update a state one step in a randomly permuted ordering of units.
//
// # Arguments
//
// * `state`: The vector to relax. Note the vector is altered in place to avoid allocating new memory.
func (network *HopfieldNetwork) UpdateState(state *mat.VecDense) {
	unitIndices := network.getUnitIndices()
	newState := mat.NewVecDense(network.dimension, nil)

	// First we must determine the (random) order of updating.
	hopfieldutils.ShuffleList(network.randomGenerator, unitIndices)
	chunkedIndices := hopfieldutils.ChunkSlice(unitIndices, network.unitsUpdatedPerStep)
	// Now we can update each index in a random order
	for _, chunk := range chunkedIndices {
		newState.MulVec(network.matrix, state)
		for _, unitIndex := range chunk {
			state.SetVec(unitIndex, newState.AtVec(unitIndex))
		}
		network.activationFunction(state)
	}
}

// Relax a state by updating until the number of unstable units is below the threshold defined by the network.
//
// # Arguments
//
// * `state`: The vector to relax. Note the vector is altered in place to avoid allocating new memory.
//
// # Returns
//
// A RelaxationResult, representing the result of relaxing a specific state.
func (network *HopfieldNetwork) RelaxState(state *mat.VecDense) *RelaxationResult {
	// We create a list of unit indices to use for randomly updating units
	unitIndices := network.getUnitIndices()
	newState := mat.NewVecDense(network.dimension, nil)

	// We will loop up to the maximum number of iterations, only returning early if the state is stable
	for stepIndex := 1; stepIndex < network.maximumRelaxationIterations; stepIndex++ {
		hopfieldutils.ShuffleList(network.randomGenerator, unitIndices)
		chunkedIndices := hopfieldutils.ChunkSlice(unitIndices, network.unitsUpdatedPerStep)
		for _, chunk := range chunkedIndices {
			newState.MulVec(network.matrix, state)
			for _, unitIndex := range chunk {
				state.SetVec(unitIndex, newState.AtVec(unitIndex))
			}
			network.activationFunction(state)
		}

		// Here we check the unit energies, counting how many unstable units there are (E>0)
		// and returning true (stable) if the number of unstable units is less than or equal to
		// the network parameter set from the builder
		if network.StateIsStable(state) {
			result := RelaxationResult{
				Stable:   true,
				NumSteps: stepIndex,
			}
			return &result
		}
	}

	// If we have reached this statement we have iterated the maximum number of times
	// and the state is STILL not stable. We return false to show the state is unstable
	result := RelaxationResult{
		Stable:   false,
		NumSteps: network.maximumRelaxationIterations,
	}
	return &result
}

// Defines a thread-orientated approach to relaxing states. Useful if the number of states to update
// is large. Note this function is intended to be used as a goroutine, i.e. go network.ConcurrentRelaxState(ch)
//
// This method uses IndexedWrappers to ensure the results of each relaxation are identifiable to the original states given.
//
// # Arguments
//
// * `stateChannel`: A channel to pass the next state to be updated to the goroutine. This channel should be created and passed before the goroutines are created.
// * `resultChannel`: A channel to pass the result of the relaxation back to the master thread.
func (network *HopfieldNetwork) concurrentRelaxStateRoutine(stateChannel chan *hopfieldutils.IndexedWrapper[*mat.VecDense], resultChannel chan *hopfieldutils.IndexedWrapper[RelaxationResult]) {
	// We create a list of unit indices to use for randomly updating units
	// Each goroutine gets a copy so they can work independently
	unitIndices := network.getUnitIndices()
	newState := mat.NewVecDense(network.dimension, nil)
	var state *mat.VecDense

	// This loop will take an indexed state from the channel until the channel is closed by the sender.
	// That is our terminating condition
	//
	// We name this loop so we can continue directly if the state is already stable.
StateRecvLoop:
	for wrappedState := range stateChannel {
		state = wrappedState.Data

		for stepIndex := 1; stepIndex < network.maximumRelaxationIterations; stepIndex++ {
			hopfieldutils.ShuffleList(network.randomGenerator, unitIndices)
			chunkedIndices := hopfieldutils.ChunkSlice(unitIndices, network.unitsUpdatedPerStep)
			for _, chunk := range chunkedIndices {
				newState.MulVec(network.matrix, state)
				for _, unitIndex := range chunk {
					state.SetVec(unitIndex, newState.AtVec(unitIndex))
				}
				network.activationFunction(state)
			}

			if network.StateIsStable(state) {
				result := RelaxationResult{
					Stable:   true,
					NumSteps: stepIndex,
				}

				resultChannel <- &hopfieldutils.IndexedWrapper[RelaxationResult]{
					Index: wrappedState.Index,
					Data:  result,
				}
				// We need not carry on with this state - continue and get the next one
				continue StateRecvLoop
			} // if unstable
		} // for iterationIndex

		// If we reach this then we did not relax correctly
		result := RelaxationResult{
			Stable:   false,
			NumSteps: network.maximumRelaxationIterations,
		}

		resultChannel <- &hopfieldutils.IndexedWrapper[RelaxationResult]{
			Index: wrappedState.Index,
			Data:  result,
		}
	}
}

// Relaxes a set of states and notes if the state is stable or not.
//
// This method works concurrently, and is the most (time) efficient way to relax a large number of states.
//
// TODO: Currently all states are dispatched before results are processed. This is done by having a large enough buffer to
// hold ALL the results. It would be MUCH better to process results in a select statement along with dispatching new states.
//
// # Arguments
//
// * `states`: A slice of states that are to be relaxed. The order of this slice corresponds to the order of the returned bools.
// * `numThreads`: An integer determining how many threads to run. Please note the master thread does not run any calculations,
// as it only dispatches states and handles results. Please check how many threads your system supports.
//
// # Returns
//
// A slice of RelaxationResult, each representing the result of relaxing a specific state.
func (network *HopfieldNetwork) ConcurrentRelaxStates(states []*mat.VecDense, numThreads int) []*RelaxationResult {
	stateChannel := make(chan *hopfieldutils.IndexedWrapper[*mat.VecDense], numThreads)
	resultChannel := make(chan *hopfieldutils.IndexedWrapper[RelaxationResult], len(states))
	results := make([]*RelaxationResult, len(states))

	// Start all the concurrent channels
	for i := 0; i < numThreads; i++ {
		go network.concurrentRelaxStateRoutine(stateChannel, resultChannel)
	}

	// var nextState hopfieldutils.IndexedWrapper[*mat.VecDense]
	for stateIndex := 0; stateIndex < len(states); stateIndex++ {
		nextState := hopfieldutils.IndexedWrapper[*mat.VecDense]{Index: stateIndex, Data: states[stateIndex]}
		stateChannel <- &nextState
	}
	close(stateChannel)

	resultsReceived := 0
	for wrappedResult := range resultChannel {
		results[wrappedResult.Index] = &wrappedResult.Data
		resultsReceived++

		if resultsReceived >= len(states) {
			break
		}
	}
	return results
}
