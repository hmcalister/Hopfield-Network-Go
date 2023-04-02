package noiseapplication

import (
	"hmcalister/hopfield/hopfieldutils"
	"math"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

// Define a function that applies noise to a given state. Note the state is affected in place.
//
// # Arguments
//
// randomGenerator *rand.Rand: A random generator used to apply noise to the state
//
// state *mat.VecDense: The state to apply noise to
//
// noiseScale float64: The amount/scale of noise to apply (if applicable).
type NoiseApplicationMethod func(*rand.Rand, *mat.VecDense, float64)

type NoiseApplicationEnum int

const (
	// Apply no noise. State is left unchanged.
	None NoiseApplicationEnum = iota

	// Multiply a number of elements of state (defined by noiseScale as a proportion of total elements) by -1
	MaximalInversion NoiseApplicationEnum = iota

	// Multiply a random number of elements of (uniform between 0 and noiseScale as proportion of total elements) by -1
	RandomSubMaximalInversion NoiseApplicationEnum = iota

	// Add an amount of Gaussian noise to all elements in the state.
	GaussianApplication NoiseApplicationEnum = iota
)

// Get a noise application function given an integer input
func GetNoiseApplicationMethod(noiseApplication NoiseApplicationEnum) NoiseApplicationMethod {
	noiseApplicationFunctions := map[NoiseApplicationEnum]NoiseApplicationMethod{
		None:                      noNoiseApplication,
		MaximalInversion:          maximalRatioInvertSliceElements,
		RandomSubMaximalInversion: randomSubMaximalRandomRatioInvertSliceElements,
		GaussianApplication:       gaussianNoise,
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
// randomGenerator *rand.Rand: A random number generator to use for selecting elements.
//
// state *mat.VecDense: The state to invert elements of
//
// inversionRatio float64: The amount of elements to invert, expressed as a ratio of the length of `state`
func maximalRatioInvertSliceElements(randomGenerator *rand.Rand, state *mat.VecDense, inversionRatio float64) {
	numInversions := int(float64(state.Len()) * inversionRatio)
	sliceIndices := make([]int, state.Len()-1)
	for i := range sliceIndices {
		sliceIndices[i] = i
	}
	hopfieldutils.ShuffleList(randomGenerator, sliceIndices)

	for i := 0; i < numInversions; i++ {
		state.SetVec(sliceIndices[i], -1*state.AtVec(sliceIndices[i]))
	}
}

// Invert a random number of  elements of a vector - up to and including n. The number of elements inverted
// is determined uniformly from 0 to n.
//
// # Arguments
//
// randomGenerator *rand.Rand: A random number generator to use for selecting elements.
//
// state *mat.VecDense: The state to invert elements of
//
// maximumInversionRatio float64: The amount of elements to invert, expressed as a ratio of the length of `state`
func randomSubMaximalRandomRatioInvertSliceElements(randomGenerator *rand.Rand, state *mat.VecDense, maximumInversionRatio float64) {
	selectedInversionRatio := math.Mod(randomGenerator.Float64(), maximumInversionRatio)
	maximalRatioInvertSliceElements(randomGenerator, state, selectedInversionRatio)
}

// Noise a given vector by applying gaussian noise to the entire vector, then applying the activation function
//
// # Arguments
//
// randomGenerator *rand.Rand: A random number generator to use for selecting elements.
//
// state *mat.VecDense: The state to apply noise to
//
// standardDeviation float64: The standard deviation of the gaussian noise to apply
func gaussianNoise(randomGenerator *rand.Rand, state *mat.VecDense, standardDeviation float64) {
	for i := 0; i < state.Len(); i++ {
		state.RawVector().Data[i] += randomGenerator.NormFloat64() * standardDeviation
	}
}
