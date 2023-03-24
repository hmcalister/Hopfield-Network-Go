package noiseapplication

import (
	"hmcalister/hopfield/hopfieldutils"
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// Define a function that applies noise to a given state.
//
// # Arguments
//
// * `randomGenerator`: A random generator used to apply noise to the state
//
// * `state`: The state to apply noise to
//
// * `noiseScale`: The amount/scale of noise to apply (if applicable).
type NoiseApplicationMethod func(*rand.Rand, *mat.VecDense, float64)

type NoiseApplicationEnum int

const (
	None                  NoiseApplicationEnum = iota
	ExactRatioInversion   NoiseApplicationEnum = iota
	UniformRatioInversion NoiseApplicationEnum = iota
	GaussianApplication   NoiseApplicationEnum = iota
)

// Get a noise application function given an integer input
func GetNoiseApplicationMethod(noiseApplication NoiseApplicationEnum) NoiseApplicationMethod {
	noiseApplicationFunctions := map[NoiseApplicationEnum]NoiseApplicationMethod{
		None:                  noNoiseApplication,
		ExactRatioInversion:   exactRatioInvertSliceElements,
	}

	return noiseApplicationFunctions[noiseApplication]
}

// Applies no noise, effectively a NOP
func noNoiseApplication(randomGenerator *rand.Rand, vec *mat.VecDense, noiseScale float64) {}

// Invert random indices of a vector. Inversion occurs by multiplying values by -1
//
// Alters the original slice in place.
//
// # Arguments
//
// * `randomGenerator`: A random number generator to use for selecting elements.
//
// * `slice`: The slice to invert elements of
//
// * `inversionRatio`: The amount of elements to invert, expressed as a ratio of the length of `slice`
func exactRatioInvertSliceElements(randomGenerator *rand.Rand, vec *mat.VecDense, inversionRatio float64) {
	numInversions := int(float64(vec.Len()) * inversionRatio)
	sliceIndices := make([]int, vec.Len()-1)
	for i := range sliceIndices {
		sliceIndices[i] = i
	}
	hopfieldutils.ShuffleList(randomGenerator, sliceIndices)

	for i := 0; i < numInversions; i++ {
		vec.SetVec(sliceIndices[i], -1*vec.AtVec(sliceIndices[i]))
	}
}

