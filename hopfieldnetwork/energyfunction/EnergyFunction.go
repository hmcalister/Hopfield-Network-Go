package energyfunction

import (
	"gonum.org/v1/gonum/mat"
)

// Defines an StateEnergyFunction as a function taking a reference to a matrix and a vector
// that returns a float64 representing the energy.
//
// This redefinition is mainly to enforce self documenting code.
type StateEnergyFunction func(*mat.Dense, *mat.VecDense) float64

// Defines UnitEnergyFunction as a function taking a reference to a matrix and a vector
// as well as an index to note the target unit that
// returns a float64 representing the energy of the unit
type UnitEnergyFunction func(*mat.Dense, *mat.VecDense, int) float64

// Defines the energy of a single unit in a vector with respect to a matrix.
func UnitEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -1 * matrix.At(i, j) * vector.AtVec(i) * vector.AtVec(j)
	}
	return energy
}

// Defines the overall energy for a state with respect to a matrix.
func StateEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	energyVector := AllUnitEnergies(matrix, vector)
	energy := 0.
	for i := 0; i < energyVector.Len(); i++ {
		energy += energyVector.AtVec(i)
	}
	return energy
}

func AllUnitEnergies(matrix *mat.Dense, vector *mat.VecDense) *mat.VecDense {
	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.MulElemVec(energyVector, vector)
	energyVector.ScaleVec(-1, energyVector)
	return energyVector
}
