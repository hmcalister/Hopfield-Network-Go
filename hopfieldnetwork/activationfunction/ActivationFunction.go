// Definitions of activation functions for the Hopfield Network
//
// Includes map from network domain to activation function
package activationfunction

import (
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"

	"gonum.org/v1/gonum/mat"
)

// Defines an ActivationFunction as a function taking a reference to a vector.
//
// This redefinition is mainly to enforce self documenting code. Note the type signature
// implies an ActivationFunction will change the vector directly - not return a new vector!
type ActivationFunction func(*mat.VecDense)

func binaryDomainMappingFunction(vector *mat.VecDense) {
	var NEGATIVE_VAL float64
	var POSITIVE_VAL float64
	if vector.AtVec(0) <= 0 {
		NEGATIVE_VAL = 1.0
		POSITIVE_VAL = 0.0
	} else {
		NEGATIVE_VAL = 0.0
		POSITIVE_VAL = 1.0
	}

	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, NEGATIVE_VAL)
		} else {
			vector.SetVec(n, POSITIVE_VAL)
		}
	}
}

func bipolarDomainMappingFunction(vector *mat.VecDense) {
	var NEGATIVE_VAL float64
	var POSITIVE_VAL float64
	if vector.AtVec(0) <= 0 {
		NEGATIVE_VAL = 1.0
		POSITIVE_VAL = -1.0
	} else {
		NEGATIVE_VAL = -1.0
		POSITIVE_VAL = 1.0
	}

	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0.0 {
			vector.SetVec(n, NEGATIVE_VAL)
		} else {
			vector.SetVec(n, POSITIVE_VAL)
		}
	}
}

func identityMappingFunction(vector *mat.VecDense) {

}

var DomainToActivationFunctionMap = map[networkdomain.NetworkDomain]ActivationFunction{
	networkdomain.BinaryDomain:     binaryDomainMappingFunction,
	networkdomain.BipolarDomain:    bipolarDomainMappingFunction,
	networkdomain.ContinuousDomain: identityMappingFunction,
}
