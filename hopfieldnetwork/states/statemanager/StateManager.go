package statemanager

import (
	"hmcalister/hopfield/hopfieldnetwork/domain"

	"gonum.org/v1/gonum/mat"
)

type StateManager interface {
	ActivationFunction(*mat.VecDense)
	InvertState(*mat.VecDense)
	UnitEnergy(*mat.Dense, *mat.VecDense, *mat.VecDense, int) float64
	AllUnitEnergies(*mat.Dense, *mat.VecDense, *mat.VecDense) []float64
	StateEnergy(*mat.Dense, *mat.VecDense, *mat.VecDense) float64
}

func GetDomainStateManager(targetDomain domain.DomainEnum) StateManager {
	domainStateManagerMap := map[domain.DomainEnum]StateManager{
		domain.BipolarDomain: &BipolarStateManager{},
		domain.BinaryDomain:  &BinaryStateManager{},
	}

	return domainStateManagerMap[targetDomain]
}
