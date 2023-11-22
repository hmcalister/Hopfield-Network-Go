package distancemeasure

import (
	"hmcalister/hopfield/hopfieldnetwork/states/domainmanager"
	"hmcalister/hopfield/hopfieldutils"

	"gonum.org/v1/gonum/mat"
)

type DistanceMeasure func(*mat.VecDense, *mat.VecDense) float64

func MeasureDistancesToCollection(collection []*mat.VecDense, a *mat.VecDense, measure DistanceMeasure) []float64 {
	distances := make([]float64, len(collection))
	for collectionIndex, currentVector := range collection {
		distances[collectionIndex] = measure(currentVector, a)
	}
	return distances
}

func GetManhattanDistance() DistanceMeasure {
	return func(a *mat.VecDense, b *mat.VecDense) float64 {
		vectorDifference := mat.NewVecDense(a.Len(), nil)
		vectorDifference.SubVec(a, b)
		return vectorDifference.Norm(1.0)
	}
}

func GetManhattanDistanceWithInversion(manager domainmanager.DomainManager) DistanceMeasure {
	return func(a *mat.VecDense, b *mat.VecDense) float64 {
		vectorDifference := mat.NewVecDense(a.Len(), nil)

		vectorDifference.SubVec(a, b)
		d1 := vectorDifference.Norm(1.0)

		aInverse := mat.VecDenseCopyOf(a)
		manager.InvertState(aInverse)
		vectorDifference.SubVec(aInverse, b)
		d2 := vectorDifference.Norm(1.0)

		return hopfieldutils.MinimumOfSlice([]float64{d1, d2})
	}
}

func GetEuclideanDistance() DistanceMeasure {
	return func(a *mat.VecDense, b *mat.VecDense) float64 {
		vectorDifference := mat.NewVecDense(a.Len(), nil)
		vectorDifference.SubVec(a, b)
		return vectorDifference.Norm(2.0)
	}
}

func GetEuclideanDistanceWithInversion(manager domainmanager.DomainManager) DistanceMeasure {
	return func(a *mat.VecDense, b *mat.VecDense) float64 {
		vectorDifference := mat.NewVecDense(a.Len(), nil)

		vectorDifference.SubVec(a, b)
		d1 := vectorDifference.Norm(2.0)

		aInverse := mat.VecDenseCopyOf(a)
		manager.InvertState(aInverse)
		vectorDifference.SubVec(aInverse, b)
		d2 := vectorDifference.Norm(2.0)

		return hopfieldutils.MinimumOfSlice([]float64{d1, d2})
	}
}
