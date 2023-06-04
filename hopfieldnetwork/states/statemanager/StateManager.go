package statemanager

import "gonum.org/v1/gonum/mat"

type StateManager interface {
	ActivationFunction(*mat.VecDense)
	InvertState(*mat.VecDense)
	UnitEnergy(*mat.Dense, *mat.VecDense, int) float64
	AllUnitEnergies(*mat.Dense, *mat.VecDense) *mat.VecDense
	StateEnergy(*mat.Dense, *mat.VecDense) float64
}
