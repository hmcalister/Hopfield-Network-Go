package hopfieldutils

import (
	"golang.org/x/exp/constraints"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

type Number interface {
	constraints.Integer | constraints.Float
}

// Find the minimum of a slice
func MinimumOfSlice[T Number](slice []T) T {
	currentMin := slice[0]
	for _, elem := range slice {
		if elem < currentMin {
			currentMin = elem
		}
	}
	return currentMin
}

// Find the maximum of a slice
func MaximumOfSlice[T Number](slice []T) T {
	currentMax := slice[0]
	for _, elem := range slice {
		if elem > currentMax {
			currentMax = elem
		}
	}
	return currentMax
}

// Shuffles the given list
func ShuffleList[T comparable](randomGenerator *rand.Rand, list []T) {
	randomGenerator.Shuffle(len(list), func(i int, j int) {
		list[i], list[j] = list[j], list[i]
	})
}

// Checks if a given element is in a slice
//
// # Arguments
//
// slice: The slice to check over
//
// elem: The element to check for
//
// # Returns
//
// True if the element is present, false otherwise
func IsInSlice[T comparable](slice []T, elem T) bool {
	for _, item := range slice {
		if item == elem {
			return true
		}
	}

	return false
}

// Defines a very simple wrapper to assign an index to another type.
//
// This type could be a *mat.VecDense to index states, or it could be an entire struct!
//
// In general an IndexedWrapper is useful to count particular types of the generic,
// such as passing items to a goroutine and tracking order before and after
type IndexedWrapper[T any] struct {
	Index int
	Data  T
}

// Allows a slice to be chunked into smaller slices. Returns a slice of slices.
//
// Note the final chunk may be smaller than chunkSize if there is a remainder upon division.
//
// # Arguments
//
// slice: The slice to chunk.
//
// chunkSize: The number of items to fit into each chunk.
//
// # Returns
//
// A slice of slices, where each internal slice (expect possibly the last one) has
// a number of items equal to chunkSize from the original list
func ChunkSlice[T any](slice []T, chunkSize int) [][]T {
	var chunkedSlices [][]T

	if chunkSize <= 0 {
		panic("chunkSize must be a positive integer!")
	}

	for i := 0; i < len(slice); i += chunkSize {
		chunkEnd := i + chunkSize

		// Ensure we do not run off the end of the array
		if chunkEnd > len(slice) {
			chunkEnd = len(slice)
		}

		chunkedSlices = append(chunkedSlices, slice[i:chunkEnd])
	}

	return chunkedSlices
}

// Measure the distance between vectors according to a norm
func MeasureDistance(vec1 *mat.VecDense, vec2 *mat.VecDense, norm float64) float64 {
	tempVec := mat.NewVecDense(vec1.Len(), nil)

	tempVec.SubVec(vec1, vec2)
	return tempVec.Norm(norm)
}

// Measure the distance between a vector and a collection of vectors
func MeasureDistancesToCollection(vectorCollection []*mat.VecDense, vec2 *mat.VecDense, norm float64) []float64 {
	tempVec := mat.NewVecDense(vec2.Len(), nil)
	distances := make([]float64, len(vectorCollection))

	for index, item := range vectorCollection {
		tempVec.SubVec(item, vec2)
		distances[index] = tempVec.Norm(norm)
	}
	return distances
}
