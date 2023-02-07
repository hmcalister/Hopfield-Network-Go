package hopfieldnetwork

import (
	"hmcalister/hopfieldnetwork/hopfieldnetwork/activationfunction"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
)

// A representation of a Hopfield Network
// Should be created using the HopfieldNetworkBuilder methods
type HopfieldNetwork struct {
	dimension             int32
	domain                networkdomain.NetworkDomain
	domainMappingFunction activationfunction.ActivationFunction
}
