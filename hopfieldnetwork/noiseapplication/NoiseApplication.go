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

// Applies no noise, effectively a NOP
func noNoiseApplication(randomGenerator *rand.Rand, vec *mat.VecDense, noiseScale float64) {}

