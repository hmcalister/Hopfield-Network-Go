package main

import (
	"hmcalister/hopfieldnetwork/hopfieldnetwork"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/states"
)

const DIMENSION int = 100
const DOMAIN networkdomain.NetworkDomain = networkdomain.BinaryDomain

// Main method for entry point
func main() {

	network := hopfieldnetwork.NewHopfieldNetworkBuilder().
		SetNetworkDimension(DIMENSION).
		SetNetworkDomain(DOMAIN).
		SetRandMatrixInit(true).
		SetMaximumRelaxationIterations(100).
		Build()

	stateGenerator := states.NewStateGeneratorBuilder().
		SetRandMin(-1).
		SetRandMax(1).
		SetGeneratorDimension(DIMENSION).
		SetGeneratorDomain(DOMAIN).
		Build()

	stableStates := stateGenerator.CreateStateCollection(50)
	network.TestHebb(stableStates)
	states := stateGenerator.CreateStateCollection(1000)
	network.ConcurrentRelaxStates(states, 1)
	// fmt.Printf("%#v", network)
}
