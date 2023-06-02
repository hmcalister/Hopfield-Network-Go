// Definitions of activation functions for the Hopfield Network
//
// Includes map from network domain to activation function
package activationfunction

import (
	"hmcalister/hopfield/hopfieldnetwork/domain"

	"gonum.org/v1/gonum/mat"
)

// Defines the activation function as a mapping that alters a vector in place
type ActivationFunction = func(*mat.VecDense)

func GetActivationFunction(domainEnum domain.DomainEnum) ActivationFunction {
	activationFunctionMap := map[domain.DomainEnum]ActivationFunction{
		domain.BipolarDomain: bipolarActivationFunction,
		domain.BinaryDomain:  binaryActivationFunction,
	}

	return activationFunctionMap[domainEnum]
}

// Apply a binary mapping to the given vector. The vector is modified in place.
// Elements of the vector set according to sign.
//
// # Arguments
//
// - vector *mat.VecDense: The vector to apply a binary mapping to.
func binaryActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, 0.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

// Apply a bipolar mapping to the given vector. The vector is modified in place.
// Elements of the vector have their values set to their sign. (0 is mapped to 1.0).
//
// # Arguments
//
// - vector *mat.VecDense: The vector to apply a bipolar mapping to.
func bipolarActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, -1.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

func continuousBipolarActivationFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) < -1.0 {
			vector.SetVec(n, -1.0)
		}
		if vector.AtVec(n) > 1.0 {
			vector.SetVec(n, 1.0)
		}
	}
}
