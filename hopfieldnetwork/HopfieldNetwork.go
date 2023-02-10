// A collection of structures and functions to implement the Hopfield Network in Go
package hopfieldnetwork

import (
	"fmt"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/activationfunction"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/energyfunction"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
	"hmcalister/hopfieldnetwork/hopfieldutils"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// A representation of a Hopfield Network.
//
// Should be created using the HopfieldNetworkBuilder methods.
type HopfieldNetwork struct {
	matrix                         *mat.Dense
	dimension                      int
	forceSymmetric                 bool
	forceZeroDiagonal              bool
	domain                         networkdomain.NetworkDomain
	maximumRelaxationUnstableUnits int
	maximumRelaxationIterations    int
	activationFunction             activationfunction.ActivationFunction
	networkEnergyFunction          energyfunction.NetworkEnergyFunction
	unitEnergyFunction             energyfunction.UnitEnergyFunction
	randomGenerator                *rand.Rand
}

// Get a reference to the weight matrix of this network
func (network HopfieldNetwork) GetMatrix() *mat.Dense {
	return network.matrix
}

// Implement Stringer for nicer formatting
func (network HopfieldNetwork) String() string {
	return fmt.Sprintf("Hopfield Network\n\tDimension: %d\n\tDomain: %s\n",
		network.dimension,
		network.domain.String())
}

// Fixes the networks matrix according to the forceSymmetric and forceZeroDiagonal properties set.
//
// This function is private as only matrix updates should need to call it.
func (network HopfieldNetwork) cleanMatrix() {
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
func (network HopfieldNetwork) getUnitIndices() []int {
	unitIndices := make([]int, network.dimension)
	for i := 0; i < network.dimension; i++ {
		unitIndices[i] = i
	}
	return unitIndices
}

// Get the energy of a given state
func (network HopfieldNetwork) StateEnergy(state *mat.VecDense) float64 {
	return network.networkEnergyFunction(network.matrix, state)
}

// Get the energy of a given unit (indexed by i) in the state
func (network HopfieldNetwork) UnitEnergy(state *mat.VecDense, i int) float64 {
	return network.unitEnergyFunction(network.matrix, state, i)
}

// Get ALL the unit energies as a slice
func (network HopfieldNetwork) AllUnitEnergies(state *mat.VecDense) []float64 {
	unitEnergies := make([]float64, network.dimension)
	for i := range unitEnergies {
		unitEnergies[i] = network.UnitEnergy(state, i)
	}
	return unitEnergies
}

// Update a state one step in a randomly permuted ordering of units
func (network HopfieldNetwork) UpdateState(state *mat.VecDense) {
	unitIndices := network.getUnitIndices()
	newState := mat.NewVecDense(network.dimension, nil)

	// First we must determine the (random) order of updating.
	hopfieldutils.ShuffleList(network.randomGenerator, unitIndices)
	// Now we can update each index in a random order
	for unitIndex := range unitIndices {
		newState.MulVec(network.matrix, state)
		network.activationFunction(newState)
		state.SetVec(unitIndex, newState.AtVec(unitIndex))
	}
}

// Relax a state by updating until the number of unstable units is below the threshold defined by the network.
func (network HopfieldNetwork) RelaxState(state *mat.VecDense) (stable bool) {
	// We create a list of unit indices to use for randomly updating units
	unitIndices := network.getUnitIndices()
	newState := mat.NewVecDense(network.dimension, nil)

	// We will loop up to the maximum number of iterations, only returning early if the state is stable
	var unstableUnits int
	for iterationIndex := 0; iterationIndex < network.maximumRelaxationIterations; iterationIndex++ {
		hopfieldutils.ShuffleList(network.randomGenerator, unitIndices)
		for unitIndex := range unitIndices {
			newState.MulVec(network.matrix, state)
			network.activationFunction(newState)
			state.SetVec(unitIndex, newState.AtVec(unitIndex))
		}

		// Here we check the unit energies, counting how many unstable units there are (E>0)
		// and returning true (stable) if the number of unstable units is less than or equal to
		// the network parameter set from the builder
		unstableUnits = 0
		unitEnergies := network.AllUnitEnergies(state)
		for _, energy := range unitEnergies {
			if energy > 0 {
				unstableUnits += 1
			}
		}
		if unstableUnits <= network.maximumRelaxationUnstableUnits {
			return true
		}
	}

	// If we have reached this statement we have iterated the maximum number of times
	// and the state is STILL not stable. We return false to show the state is unstable
	return false
}

// Defines a thread-orientated approach to relaxing states. Useful if the number of states to update
// is large. Note this function is intended to be used as a goroutine, i.e. go network.ConcurrentRelaxState(ch)
//
// stateChannel is a channel to pass the next state to be updated to the goroutine. This channel should be
// created and passed before the goroutines are created.
func (network HopfieldNetwork) concurrentRelaxStateRoutine(stateChannel chan *hopfieldutils.IndexedWrapper[*mat.VecDense], resultChannel chan *hopfieldutils.IndexedWrapper[bool]) {
	// We create a list of unit indices to use for randomly updating units
	// Each goroutine gets a copy so they can work independently
	unitIndices := network.getUnitIndices()
	newState := mat.NewVecDense(network.dimension, nil)
	var unstableUnits int
	var currentState *mat.VecDense

	// This loop will take an indexed state from the channel until the channel is closed by the sender.
	// That is our terminating condition
	//
	// We name this loop so we can continue directly if the state is already stable.
StateRecvLoop:
	for currentStateWrapped := range stateChannel {
		currentState = currentStateWrapped.Data

		for iterationIndex := 0; iterationIndex < network.maximumRelaxationIterations; iterationIndex++ {
			hopfieldutils.ShuffleList(network.randomGenerator, unitIndices)
			for unitIndex := range unitIndices {
				newState.MulVec(network.matrix, currentState)
				network.activationFunction(newState)
				currentState.SetVec(unitIndex, newState.AtVec(unitIndex))
			}

			// Here we check the unit energies, counting how many unstable units there are (E>0)
			// and returning true (stable) if the number of unstable units is less than or equal to
			// the network parameter set from the builder
			unstableUnits = 0
			unitEnergies := network.AllUnitEnergies(currentState)
			for _, energy := range unitEnergies {
				if energy > 0 {
					unstableUnits += 1
				}
			}

			// If we have a stable enough state (within unstable units tolerance)
			// send back true (stable) along with the index
			if unstableUnits <= network.maximumRelaxationUnstableUnits {
				resultChannel <- &hopfieldutils.IndexedWrapper[bool]{
					Index: currentStateWrapped.Index,
					Data:  true,
				}
				// We need not carry on with this state - continue and get the next one
				continue StateRecvLoop
			} // if unstable
		} // for iterationIndex

		// If we reach this
		resultChannel <- &hopfieldutils.IndexedWrapper[bool]{
			Index: currentStateWrapped.Index,
			Data:  true,
		}
	}
}

// Relaxes a set of states and notes if the state is stable or not
func (network HopfieldNetwork) ConcurrentRelaxStates(states []*mat.VecDense, numThreads int) {
	stateChannel := make(chan *hopfieldutils.IndexedWrapper[*mat.VecDense], 10)
	resultChannel := make(chan *hopfieldutils.IndexedWrapper[bool], len(states))
	results := make([]bool, len(states))

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
	numStable := 0
	for wrappedResult := range resultChannel {
		results[wrappedResult.Index] = wrappedResult.Data
		resultsReceived++

		if wrappedResult.Data {
			numStable++
		}

		if resultsReceived >= len(states) {
			break
		}
	}
	fmt.Println(numStable)
}

func (network HopfieldNetwork) TestHebb(states []*mat.VecDense) {
	for _, state := range states {
		for i := 0; i < network.dimension; i++ {
			for j := 0; j < network.dimension; j++ {
				val := (2*state.AtVec(i) - 1) * (2*state.AtVec(j) - 1)
				val += network.matrix.At(i, j)
				network.matrix.Set(i, j, val)
			}
		}
	}
}
