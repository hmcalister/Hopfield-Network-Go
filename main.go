package main

import (
	"fmt"
	"hmcalister/hopfield/hopfieldnetwork"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
	"hmcalister/hopfield/hopfieldnetwork/states"
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

	states := stateGenerator.CreateStateCollection(1000)
	relaxResults := network.ConcurrentRelaxStates(states, 10)
	numStable := 0
	for _, r := range relaxResults {
		if r {
			numStable++
		}
	}
	fmt.Println(numStable)
}
