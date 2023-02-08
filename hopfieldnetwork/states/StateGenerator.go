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

// Create a new random state, allocating new memory using make.
//
// Note this function may be slow compared to using CreateStateMemory and NextStateToMemory.
func (gen *StateGenerator) NextState() *mat.VecDense {
	stateData := make([]float64, gen.dimension)
	for i := range stateData {
		stateData[i] = gen.rng.Rand()
	}

	state := mat.NewVecDense(gen.dimension, stateData)
	gen.activationFunction(state)
	return state
}
