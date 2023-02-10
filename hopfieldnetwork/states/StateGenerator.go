// A package to handle generating states for the Hopfield Network
package states

import (
	"hmcalister/hopfieldnetwork/hopfieldnetwork/activationfunction"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Define a struct that can generate new states.
//
// Note this struct should be initialized using the StateGeneratorBuilder from [hmcalister/hopfieldnetwork/hopfieldnetwork/states].
type StateGenerator struct {
	rng                distuv.Uniform
	dimension          int
	activationFunction activationfunction.ActivationFunction
}

// Creates and returns a fresh array that can store a state.
//
// This is implemented for cleaner, more explicit code. Rather than calling NextState once to get memory,
// this method will create the memory independently. For example, calling AllocStateMemory before a loop and passing
// the result to NextState within the loop looks a lot nicer than some messy, potentially nil value being passed about.
func (gen *StateGenerator) AllocStateMemory() []float64 {
	return make([]float64, gen.dimension)
}

// Create a new random state, using dataArray to store intermediate random values.
//
// Note this function require an array to be allocated already. Don't panic! Use AllocDataArray to get this memory!
func (gen *StateGenerator) NextState(dataArray []float64) *mat.VecDense {
	for i := 0; i < gen.dimension; i++ {
		dataArray[i] = gen.rng.Rand()
	}

	state := mat.NewVecDense(gen.dimension, dataArray)
	gen.activationFunction(state)
	return state
}

// Creates a set of new states, returning the new states as a pointer to a slice of VecDense
//
// Note this function does NOT require new memory to be allocated - it is allocated in the method.
func (gen *StateGenerator) CreateStateCollection(numStates int) []*mat.VecDense {
	states := make([]*mat.VecDense, numStates)

	var backingMem []float64
	for i := range states {
		backingMem = gen.AllocStateMemory()
		states[i] = gen.NextState(backingMem)
	}
	return states
}
