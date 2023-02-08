// A collection of structures and functions to implement the Hopfield Network in Go
package hopfieldnetwork

import (
	"hmcalister/hopfieldnetwork/hopfieldnetwork/activationfunction"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"

	"gonum.org/v1/gonum/mat"
)

// A representation of a Hopfield Network.
//
// Should be created using the HopfieldNetworkBuilder methods.
type HopfieldNetwork struct {
	matrix             *mat.Dense
	dimension          int
	domain             networkdomain.NetworkDomain
	activationFunction activationfunction.ActivationFunction
}
