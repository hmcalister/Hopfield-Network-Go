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
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0 {
			vector.SetVec(n, 0)
		} else {
			vector.SetVec(n, 1)
		}
	}
}

func bipolarDomainMappingFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) <= 0 {
			vector.SetVec(n, -1)
		} else {
			vector.SetVec(n, 1)
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
