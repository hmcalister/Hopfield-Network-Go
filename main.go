package main

import (
	"fmt"
	"hmcalister/hopfieldnetwork/hopfieldnetwork"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/states"
)

// Main method for entry point
func main() {

	builder := hopfieldnetwork.NewHopfieldNetworkBuilder().
		SetNetworkDimension(100).
		SetNetworkDomain(networkdomain.BinaryDomain)

	network := builder.Build()
	fmt.Printf("%#v\n", network)

	stateGenBuilder := states.NewStateGeneratorBuilder().
		SetRandMin(-10).
		SetRandMax(1).
		SetGeneratorDimension(100).
		SetGeneratorDomain(networkdomain.BinaryDomain)
	s1 := stateGenBuilder.Build()
	s2 := stateGenBuilder.Build()
	fmt.Printf("%#v\n%#v\n", s1, s2)

	fmt.Printf("%#v\n", s1.NextState())
}
