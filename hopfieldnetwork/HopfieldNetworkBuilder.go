package hopfieldnetwork

import (
	"hmcalister/hopfieldnetwork/hopfieldnetwork/activationfunction"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
)

type HopfieldNetworkBuilder struct {
	dimension int
	domain    networkdomain.NetworkDomain
}

// Get a new HopfieldNetworkBuilder filled with the default values.
//
// Note that some default values will cause build errors - this is intentional!
// Users should explicitly set at least these values before building.
func NewHopfieldNetworkBuilder() *HopfieldNetworkBuilder {
	return &HopfieldNetworkBuilder{
		dimension: 0,
		domain:    networkdomain.UnspecifiedDomain,
	}
}

// Set the domain of the HopfieldNetwork - i.e. what numbers are allowed to exist in states.
//
// Valid options are taken from the NetworkDomain enum (BinaryDomain, BipolarDomain, ContinuousDomain).
// Note that UnspecifiedDomain is the default and throws and error if building is attempted.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be specified before Build can be called.
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkDomain(domain networkdomain.NetworkDomain) *HopfieldNetworkBuilder {
	networkBuilder.domain = domain
	return networkBuilder
}

// Set the dimension of the HopfieldNetwork - i.e. the dimension of the square matrix.
//
// Note this method returns the builder pointer so chained calls can be used.
//
// Must be set specified Build can be called
func (networkBuilder *HopfieldNetworkBuilder) SetNetworkDimension(dimension int) *HopfieldNetworkBuilder {
	networkBuilder.dimension = dimension
	return networkBuilder
}

// Build and return a new HopfieldNetwork using the parameters specified with builder methods.
func (networkBuilder *HopfieldNetworkBuilder) Build() HopfieldNetwork {
	if networkBuilder.dimension <= 0 {
		panic("HopfieldNetworkBuilder encountered an error during build! Dimension must be explicitly set to a positive integer!")
	}

	if networkBuilder.domain == networkdomain.UnspecifiedDomain {
		panic("HopfieldNetworkBuilder encountered an error during build! Domain must be explicitly set to a valid network domain!")
	}

	activationFunction := activationfunction.DomainToActivationFunctionMap[networkBuilder.domain]

	return HopfieldNetwork{
		dimension:          networkBuilder.dimension,
		domain:             networkBuilder.domain,
		activationFunction: activationFunction,
	}

}
