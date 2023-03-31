package hopfieldutils

import (
	"errors"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Save a matrix to the given file.
//
// If the file does not exist, attempts to create it.
// Warning: this method will overwrite the file it is pointed at!
//
// # Arguments
//
// *`matrix`: The matrix to save
//
// *`savepath`: The path to the file to save into
//
// # Returns
//
// Error if something went wrong, nil if saved correctly
func SaveMatrix(matrix *mat.Dense, savepath string) error {
	f, err := os.OpenFile(savepath, os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = matrix.MarshalBinaryTo(f)
	if err != nil {
		return err
	}

	return nil
}

// Loads a matrix from a saved file
//
// # Arguments
//
// *`savepath`: The path to the file the matrix is saved in
//
// # Returns
//
//	(loaded matrix, nil) on success, (nil, error) on errors
func LoadMatrix(savepath string) (*mat.Dense, error) {
	f, err := os.OpenFile(savepath, os.O_RDONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	matrix := mat.Dense{}
	if _, err := matrix.UnmarshalBinaryFrom(f); err != nil {
		return nil, err
	}

	return &matrix, nil
}

// Save a vector to the given file.
//
// If the file does not exist, attempts to create it.
// Warning: this method will overwrite the file it is pointed at!
//
// # Arguments
//
// *`vector`: The vector to save
//
// *`savepath`: The path to the file to save into
//
// # Returns
//
// Error if something went wrong, nil if saved correctly
func SaveVector(vec *mat.VecDense, savepath string) error {
	f, err := os.OpenFile(savepath, os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = vec.MarshalBinaryTo(f)
	if err != nil {
		return err
	}

	return nil
}

// Loads a vector from a saved file
//
// # Arguments
//
// *`savepath`: The path to the file the vector is saved in
//
// # Returns
//
//	(loaded vector, nil) on success, (nil, error) on errors
func LoadVector(savepath string) (*mat.VecDense, error) {
	f, err := os.OpenFile(savepath, os.O_RDONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	vec := mat.NewVecDense(1, nil)
	if _, err := vec.UnmarshalBinaryFrom(f); err != nil {
		return nil, err
	}

	return vec, nil
}

// Save a vector collection to the given file.
//
// If the file does not exist, attempts to create it.
// Warning: this method will overwrite the file it is pointed at!
//
// This method first organizes all the given vectors as rows of a matrix before saving the matrix.
// Be aware this requires all vectors in the collection to have the same length!
//
// # Arguments
//
// *`vecCollection`: The vector collection to save
//
// *`savepath`: The path to the file to save into
//
// # Returns
//
// Error if something went wrong, nil if saved correctly
func SaveVectorCollection(vecCollection []*mat.VecDense, savepath string) error {
	if len(vecCollection) == 0 {
		return errors.New("vector collection is empty")
	}

	// Ensure vectors have same length
	vecLength := vecCollection[0].Len()
	for _, vec := range vecCollection {
		if vec.Len() != vecLength {
			return errors.New("vector collection contains vectors of mismatched length")
		}
	}

	f, err := os.OpenFile(savepath, os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	// Append all the collection data into a single array
	allData := []float64{}
	for _, vec := range vecCollection {
		allData = append(allData, vec.RawVector().Data...)
	}

	// Form that array into a single matrix and save
	vectorMatrix := mat.NewDense(len(vecCollection), vecCollection[0].Len(), allData)

	_, err = vectorMatrix.MarshalBinaryTo(f)
	if err != nil {
		return err
	}

	return nil
}

// Loads a vectorCollection from a saved file
//
// # Arguments
//
// *`savepath`: The path to the file the vectorCollection is saved in
//
// # Returns
//
//	(loaded []*mat.VecDense, nil) on success, (nil, error) on errors
func LoadVectorCollection(savepath string) ([]*mat.VecDense, error) {
	f, err := os.OpenFile(savepath, os.O_RDONLY|os.O_CREATE, 0644)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	matrix, err := LoadMatrix(savepath)
	if err != nil {
		return nil, err
	}

	// Destructure matrix back into vectors
	vecCollection := make([]*mat.VecDense, matrix.RawMatrix().Rows)

	for rowIndex := 0; rowIndex < matrix.RawMatrix().Rows; rowIndex++ {
		vecCollection[rowIndex] = mat.VecDenseCopyOf(matrix.RowView(rowIndex))
	}

	return vecCollection, nil
}
