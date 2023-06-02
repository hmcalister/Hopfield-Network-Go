package domain

// An enum to note the domain of the network.
// This determines the learning mapping function to use.
type DomainEnum int

const (
	BipolarDomain DomainEnum = iota
	BinaryDomain  DomainEnum = iota
)
