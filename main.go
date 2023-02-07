package main

import (
	"fmt"
	"hmcalister/hopfieldnetwork/hopfieldnetwork"
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"
)

func main() {

	builder := hopfieldnetwork.NewHopfieldNetworkBuilder().
		SetNetworkDimension(100).
		SetNetworkDomain(networkdomain.BipolarDomain)

	network := builder.Build()
	fmt.Printf("%#v\n", network)
}
