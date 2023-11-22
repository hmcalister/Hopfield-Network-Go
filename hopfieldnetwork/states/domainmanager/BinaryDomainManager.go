package domainmanager

import "gonum.org/v1/gonum/mat"

type BinaryDomainManager struct {
}

func (manager *BinaryDomainManager) mapVectorToBipolar(vector *mat.VecDense) *mat.VecDense {
	negativeOnesVector := manager.createCompatibleConstVector(vector, -1.0)
	mappedVector := mat.VecDenseCopyOf(vector)
	mappedVector.AddScaledVec(negativeOnesVector, 2.0, mappedVector)
	return mappedVector
}

func (manager *BinaryDomainManager) createCompatibleConstVector(origVector *mat.VecDense, vectorConst float64) *mat.VecDense {
	constVector := mat.NewVecDense(origVector.Len(), nil)
	for i := 0; i < constVector.Len(); i++ {
		constVector.SetVec(i, vectorConst)
	}
	return constVector
}

func (manager *BinaryDomainManager) ActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, 0.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

func (manager *BinaryDomainManager) InvertState(vector *mat.VecDense) {
	onesVector := manager.createCompatibleConstVector(vector, 1.0)
	vector.AddScaledVec(onesVector, -1.0, vector)
	manager.ActivationFunction(vector)
}

func (manager *BinaryDomainManager) UnitEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	mappedVector := manager.mapVectorToBipolar(vector)
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -0.5*matrix.At(i, j)*vector.AtVec(i)*mappedVector.AtVec(j) - 1
	}

	return energy
}

func (manager *BinaryDomainManager) AllUnitEnergies(matrix *mat.Dense, vector *mat.VecDense) []float64 {
	mappedVector := manager.mapVectorToBipolar(vector)

	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.MulElemVec(energyVector, mappedVector)
	energyVector.ScaleVec(-0.5, energyVector)

	return energyVector.RawVector().Data
}

func (manager *BinaryDomainManager) StateEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	energyVector := manager.AllUnitEnergies(matrix, vector)
	energy := 0.0
	for _, unitEnergy := range energyVector {
		energy += unitEnergy
	}
	return energy
}
