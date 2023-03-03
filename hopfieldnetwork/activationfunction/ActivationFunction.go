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

func unitSphereMappingFunction(vector *mat.VecDense) {
	norm := vector.Norm(2.0)
	vector.ScaleVec(1/norm, vector)
}

func continuousCubeMappingFunction(vector *mat.VecDense) {
	for n := 0; n < vector.Len(); n++ {
		if vector.AtVec(n) < 0.0 {
			vector.SetVec(n, -1.0)
		}
		if vector.AtVec(n) > 1.0 {
			vector.SetVec(n, 1.0)
		}
	}
}

var domainToActivationFunctionMap = map[networkdomain.NetworkDomain]ActivationFunction{
	networkdomain.BipolarDomain:  bipolarDomainMappingFunction,
	networkdomain.UnitSphere:     unitSphereMappingFunction,
	networkdomain.ContinuousCube: continuousCubeMappingFunction,
}

// Given a domain, get the activation function the network will use during relaxing.
//
// # Arguments
//
// * `domain`: The network domain.
//
// # Returns
//
// The activation function to use during relaxing a state.
func GetNetworkActivationFunction(domain networkdomain.NetworkDomain) ActivationFunction {
	return domainToActivationFunctionMap[domain]
}

// Given a domain, get the activation function used to create new states.
//
// # Arguments
//
// * `domain`: The network domain.
//
// # Returns
//
// The activation function to create a new state.
func GetGeneralStateMapper(domain networkdomain.NetworkDomain) ActivationFunction {
	domainToGeneralStateMapper := map[networkdomain.NetworkDomain]ActivationFunction{
		networkdomain.ContinuousCube: continuousCubeMappingFunction,
	}
	activationFn, ok := domainToGeneralStateMapper[domain]

	if ok {
		return activationFn
	} else {
		// If the network domain is not specified here, use the general activation instead
		return domainToActivationFunctionMap[domain]
	}

}

// Given a domain, get the activation function used to create learned states.
//
// # Arguments
//
// * `domain`: The network domain.
//
// # Returns
//
// The activation function to create a new state.
func GetLearnedStateMapper(domain networkdomain.NetworkDomain) ActivationFunction {
	domainToGeneralStateMapper := map[networkdomain.NetworkDomain]ActivationFunction{
		networkdomain.ContinuousCube: bipolarDomainMappingFunction,
	}
	activationFn, ok := domainToGeneralStateMapper[domain]

	if ok {
		return activationFn
	} else {
		// If the network domain is not specified here, use the general activation instead
		return GetGeneralStateMapper(domain)
	}

}
