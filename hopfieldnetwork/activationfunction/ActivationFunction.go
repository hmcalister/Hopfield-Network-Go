// Definitions of activation functions for the Hopfield Network
//
// Includes map from network domain to activation function
package activationfunction

import (
	"gonum.org/v1/gonum/mat"
)

// Defines the activation to be used in the simulation.
//
// TODO(hayden): In future, consider a command line flag to set this.
var ActivationFunction = binaryDomainMappingFunction

// Apply a binary mapping to the given vector. The vector is modified in place.
// Elements of the vector set according to sign.
//
// # Arguments
//
// - vector *mat.VecDense: The vector to apply a binary mapping to.
func binaryDomainMappingFunction(vector *mat.VecDense) {
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
func bipolarDomainMappingFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) < 0.0 {
			vector.SetVec(n, -1.0)
		} else {
			vector.SetVec(n, 1.0)
		}
	}
}

func continuousBipolarDomainMappingFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) < -1.0 {
			vector.SetVec(n, -1.0)
		}
		if vector.AtVec(n) > 1.0 {
			vector.SetVec(n, 1.0)
		}
	}
}
