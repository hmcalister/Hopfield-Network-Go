package domain

import (
	"gonum.org/v1/gonum/mat"
)

type DomainManager interface {
	ActivationFunction(*mat.VecDense)
	ActivationFunctionUnit(float64) float64
	InvertState(*mat.VecDense)
	UnitEnergy(*mat.Dense, *mat.VecDense, int) float64
	AllUnitEnergies(*mat.Dense, *mat.VecDense) []float64
	StateEnergy(*mat.Dense, *mat.VecDense) float64
}

func GetDomainManager(targetDomain DomainEnum) DomainManager {
	domainStateManagerMap := map[DomainEnum]DomainManager{
		BipolarDomain: &BipolarDomainManager{},
		BinaryDomain:  &BinaryDomainManager{},
	}

	return domainStateManagerMap[targetDomain]
}
