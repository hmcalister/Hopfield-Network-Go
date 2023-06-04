package energyfunction

import (
	"gonum.org/v1/gonum/mat"
)

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
	energy := 0.0
	for i := 0; i < energyVector.Len(); i++ {
		energy += energyVector.AtVec(i)
	}
	return energy
}

// Gets all the unit energies of a given vector with respect to a matrix.
func AllUnitEnergies(matrix *mat.Dense, vector *mat.VecDense) *mat.VecDense {
	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.MulElemVec(energyVector, vector)
	energyVector.ScaleVec(-1, energyVector)
	return energyVector
}
