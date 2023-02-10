package energyfunction

import (
	"hmcalister/hopfieldnetwork/hopfieldnetwork/networkdomain"

	"gonum.org/v1/gonum/mat"
)

// Defines an NetworkEnergyFunction as a function taking a reference to a matrix and a vector
// that returns a float64 representing the energy.
//
// This redefinition is mainly to enforce self documenting code.
type NetworkEnergyFunction func(*mat.Dense, *mat.VecDense) float64

// Defines UnitEnergyFunction as a function taking a reference to a matrix and a vector
// as well as an index to note the target unit that
// returns a float64 representing the energy of the unit
type UnitEnergyFunction func(*mat.Dense, *mat.VecDense, int) float64

// Defines the UnitEnergy for a binary network
func UnitBinaryEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -1 * matrix.At(i, j) * vector.AtVec(i) * vector.AtVec(j)
	}
	return energy
}

// Defines the overall network energy for a binary network
func NetworkBinaryEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for i := 0; i < dimension; i++ {
		energy += UnitBinaryEnergy(matrix, vector, i)
	}
	return energy
}

// Defines the UnitEnergy for a binary network
func UnitBipolarEnergy(matrix *mat.Dense, vector *mat.VecDense, i int) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for j := 0; j < dimension; j++ {
		energy += -1 * matrix.At(i, j) * vector.AtVec(i) * vector.AtVec(j)
	}
	return energy
}

// Defines the overall network energy for a binary network
func NetworkBipolarEnergy(matrix *mat.Dense, vector *mat.VecDense) float64 {
	dimension, _ := vector.Dims()
	energy := 0.0
	for i := 0; i < dimension; i++ {
		energy += UnitBinaryEnergy(matrix, vector, i)
	}
	return energy
}

var DomainToNetworkEnergyFunctionMap = map[networkdomain.NetworkDomain]NetworkEnergyFunction{
	networkdomain.BinaryDomain:  NetworkBinaryEnergy,
	networkdomain.BipolarDomain: NetworkBipolarEnergy,
}

var DomainToUnitEnergyFunctionMap = map[networkdomain.NetworkDomain]UnitEnergyFunction{
	networkdomain.BinaryDomain:  UnitBinaryEnergy,
	networkdomain.BipolarDomain: UnitBipolarEnergy,
}
