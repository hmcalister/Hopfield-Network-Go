package statemanager

import "gonum.org/v1/gonum/mat"

type BipolarStateManager struct {
}

func (manager *BipolarStateManager) ActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, -1.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

func (manager *BipolarStateManager) InvertState(vector *mat.VecDense) {
	vector.ScaleVec(-1.0, vector)
	manager.ActivationFunction(vector)
}

func (manager *BipolarStateManager) UnitEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -1 * matrix.At(i, j) * vector.AtVec(i) * vector.AtVec(j)
	}
	return energy
}

func (manager *BipolarStateManager) AllUnitEnergies(matrix *mat.Dense, vector *mat.VecDense) []float64 {
	energyVector := mat.NewVecDense(vector.Len(), nil)
	energyVector.MulVec(matrix, vector)
	energyVector.MulElemVec(energyVector, vector)
	energyVector.ScaleVec(-1, energyVector)
	return energyVector.RawVector().Data
}

func (manager *BipolarStateManager) StateEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	energyVector := manager.AllUnitEnergies(matrix, vector)
	energy := 0.0
	for _, unitEnergy := range energyVector {
		energy += unitEnergy
	}
	return energy
}

func (manager *BipolarStateManager) MeasureDistance(vec1 *mat.VecDense, vec2 *mat.VecDense, norm float64) float64 {
	tempVec := mat.NewVecDense(vec1.Len(), nil)

	tempVec.SubVec(vec1, vec2)
	d1 := tempVec.Norm(norm)

	tempVec.CopyVec(vec2)
	manager.InvertState(tempVec)
	tempVec.SubVec(vec1, tempVec)
	d2 := tempVec.Norm(norm)

	if d1 < d2 {
		return d1
	} else {
		return d2
	}
}

func (manager *BipolarStateManager) MeasureDistancesToCollection(vectorCollection []*mat.VecDense, vec2 *mat.VecDense, norm float64) []float64 {
	tempVec := mat.NewVecDense(vec2.Len(), nil)
	distances := make([]float64, len(vectorCollection))

	for index, item := range vectorCollection {
		tempVec.SubVec(item, vec2)
		d1 := tempVec.Norm(norm)

		tempVec.CopyVec(vec2)
		manager.InvertState(tempVec)
		tempVec.SubVec(item, tempVec)
		d2 := tempVec.Norm(norm)

		if d1 < d2 {
			distances[index] = d1
		} else {
			distances[index] = d2
		}
	}
	return distances
}
