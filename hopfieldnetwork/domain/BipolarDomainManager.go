package domain

import (
	"gonum.org/v1/gonum/mat"
)

type BipolarDomainManager struct {
}

func (manager *BipolarDomainManager) ActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, -1.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

func (manager *BipolarDomainManager) InvertState(vector *mat.VecDense) {
	vector.ScaleVec(-1.0, vector)
	manager.ActivationFunction(vector)
}

func (manager *BipolarDomainManager) UnitEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -0.5 * matrix.At(i, j) * vector.AtVec(i) * vector.AtVec(j)
	}

	return energy
}

func (manager *BipolarDomainManager) AllUnitEnergies(matrix *mat.Dense, vector *mat.VecDense) []float64 {
	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.MulElemVec(energyVector, vector)
	energyVector.ScaleVec(-0.5, energyVector)

	return energyVector.RawVector().Data
}

func (manager *BipolarDomainManager) StateEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	energyVector := manager.AllUnitEnergies(matrix, vector)
	energy := 0.0
	for _, unitEnergy := range energyVector {
		energy += unitEnergy
	}
	return energy
}
