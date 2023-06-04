package statemanager

import "gonum.org/v1/gonum/mat"

type BinaryStateManager struct {
}

func (binaryManager *BinaryStateManager) createCompatibleConstVector(origVector *mat.VecDense, vectorConst float64) *mat.VecDense {
	constVector := mat.NewVecDense(origVector.Len(), nil)
	for i := 0; i < constVector.Len(); i++ {
		constVector.SetVec(i, vectorConst)
	}
	return constVector
}

func (manager *BinaryStateManager) ActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, 0.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

func (manager *BinaryStateManager) InvertState(vector *mat.VecDense) {
	onesVector := manager.createCompatibleConstVector(vector, 1.0)
	vector.AddScaledVec(onesVector, -1.0, vector)
	manager.ActivationFunction(vector)
}

// TODO: ENERGIES

func (manager *BinaryStateManager) UnitEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -1 * matrix.At(i, j) * (2*vector.AtVec(i) - 1) * (2*vector.AtVec(j) - 1)
	}
	return energy
}

func (manager *BinaryStateManager) AllUnitEnergies(matrix *mat.Dense, vector *mat.VecDense) []float64 {
	negativeOnesVector := manager.createCompatibleConstVector(vector, -1.0)
	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.AddScaledVec(negativeOnesVector, 2.0, energyVector)
	energyVector.MulElemVec(energyVector, vector)
	energyVector.ScaleVec(-1.0, energyVector)
	return energyVector.RawVector().Data
}

func (manager *BinaryStateManager) StateEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	energyVector := manager.AllUnitEnergies(matrix, vector)
	energy := 0.0
	for _, unitEnergy := range energyVector {
		energy += unitEnergy
	}
	return energy
}
