package learningmappingfunction

import (
	"hmcalister/hopfield/hopfieldnetwork/domain"

	"gonum.org/v1/gonum/mat"
)

// A learningMappingFunction is a map taking a state vector
// (specifically a target state vector) and maps it into a domain
// that the learning rule can be applied to. For example, the
// bipolar domain requires no mapping as the learning rules already
// support this directly, while the binary domain needs to be mapped
// into the bipolar domain.
type LearningMappingFunction func(*mat.VecDense)

func GetLearningMappingFunction(domainEnum domain.DomainEnum) LearningMappingFunction {
	learningMappingFunctionMap := map[domain.DomainEnum]LearningMappingFunction{
		domain.BipolarDomain: bipolarLearningMappingFunction,
		domain.BinaryDomain:  binaryLearningMappingFunction,
	}

	return learningMappingFunctionMap[domainEnum]
}

func binaryLearningMappingFunction(vector *mat.VecDense) {
	negativeOneVector := mat.NewVecDense(vector.Len(), nil)
	for i := 0; i < negativeOneVector.Len(); i++ {
		negativeOneVector.SetVec(i, -1.0)
	}
	vector.AddScaledVec(negativeOneVector, 2, vector)
}

func bipolarLearningMappingFunction(vector *mat.VecDense) {

}
