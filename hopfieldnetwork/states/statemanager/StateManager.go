package statemanager

import (
	"hmcalister/hopfield/hopfieldnetwork/domain"

	"gonum.org/v1/gonum/mat"
)

type StateManager interface {
	ActivationFunction(*mat.VecDense)
	InvertState(*mat.VecDense)
	UnitEnergy(*mat.Dense, *mat.VecDense, int) float64
	AllUnitEnergies(*mat.Dense, *mat.VecDense) []float64
	StateEnergy(*mat.Dense, *mat.VecDense) float64

	// Finds the distance between two vectors over a domain
	MeasureDistance(*mat.VecDense, *mat.VecDense, float64) float64

	// Finds the distances from a given vector to all vectors in a slice
	//
	// # Arguments
	//
	// vectorCollection: The collection to check over
	//
	// vector: The element to check the distance to
	//
	// norm: The norm function to use. 1 for Hamming Distance, 2 for Euclidean...
	//
	// # Return
	//
	// A []float64 representing the distances from the given vector to the slice vectors
	MeasureDistancesToCollection([]*mat.VecDense, *mat.VecDense, float64) []float64
}

func GetDomainStateManager(targetDomain domain.DomainEnum) StateManager {
	domainStateManagerMap := map[domain.DomainEnum]StateManager{
		domain.BipolarDomain: &BipolarStateManager{},
		domain.BinaryDomain:  &BinaryStateManager{},
	}

	return domainStateManagerMap[targetDomain]
}
