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

var domainToActivationFunctionMap = map[networkdomain.NetworkDomain]ActivationFunction{
	networkdomain.BipolarDomain: bipolarDomainMappingFunction,
}

// Given a domain, get the activation function to map an arbitrary vector onto that domain.
//
// # Arguments
//
// * `domain`: The network domain.
//
// # Returns
//
// The activation function to map a vector to that domain.
func GetDomainActivationFunction(domain networkdomain.NetworkDomain) ActivationFunction {
	return domainToActivationFunctionMap[domain]
}
