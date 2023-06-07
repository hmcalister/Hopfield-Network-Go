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

func (manager *BinaryStateManager) UnitEnergy(matrix *mat.Dense, bias *mat.VecDense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -0.5 * matrix.At(i, j) * vector.AtVec(i) * (2*vector.AtVec(j) - 1)
	}
	energy += -1.0 * vector.AtVec(i) * bias.AtVec(i)

	return energy
}

func (manager *BinaryStateManager) AllUnitEnergies(matrix *mat.Dense, bias *mat.VecDense, vector *mat.VecDense) []float64 {
	negativeOnesVector := manager.createCompatibleConstVector(vector, -1.0)
	mappedVector := mat.VecDenseCopyOf(vector)
	mappedVector.AddScaledVec(negativeOnesVector, 2.0, mappedVector)

	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.MulElemVec(energyVector, mappedVector)
	energyVector.ScaleVec(-0.5, energyVector)
	energyVector.AddScaledVec(energyVector, -1.0, bias)
	return energyVector.RawVector().Data
}

func (manager *BinaryStateManager) StateEnergy(matrix *mat.Dense, bias *mat.VecDense, vector *mat.VecDense) float64 {
	energyVector := manager.AllUnitEnergies(matrix, bias, vector)
	energy := 0.0
	for _, unitEnergy := range energyVector {
		energy += unitEnergy
	}
	return energy
}
