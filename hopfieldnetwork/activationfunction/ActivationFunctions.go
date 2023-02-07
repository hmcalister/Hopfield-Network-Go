package activationfunction

import (
	"gonum.org/v1/gonum/mat"

	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
)

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

func continuousDomainMappingFunction(vector *mat.VecDense) {

}

var DomainToActivationFunctionMap = map[networkdomain.NetworkDomain]ActivationFunction{
	networkdomain.BinaryDomain:     binaryDomainMappingFunction,
	networkdomain.BipolarDomain:    bipolarDomainMappingFunction,
	networkdomain.ContinuousDomain: continuousDomainMappingFunction,
}
