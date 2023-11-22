package states

import (
	"hmcalister/hopfield/hopfieldnetwork/domain"
	"time"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/stat/distuv"
)

// Define a builder for a new state generator.
//
// rand_min defines the lower bound on the uniform distribution to use for state generation.
//
// rand_max defines the upper bound on the uniform distribution to use for state generation.
//
// Seed defines the seed to use for the new uniform distribution. The default value is 0.
// If seed is 0 build time then a random seed is selected. This is useful for having different threads use different seeds.
//
// Dimension defines the length of the vector to be generated.
type StateGeneratorBuilder struct {
	randMin       float64
	randMax       float64
	seed          uint64
	seedGenerator *rand.Rand
	domain        domain.DomainEnum
	dimension     int
}

func NewStateGeneratorBuilder() *StateGeneratorBuilder {
	seedGenSrc := rand.NewSource(uint64(time.Now().Nanosecond()))
	seedGen := rand.New(seedGenSrc)
	return &StateGeneratorBuilder{
		randMin:       -1,
		randMax:       1,
		seed:          0,
		seedGenerator: seedGen,
		domain:        0,
		dimension:     0,
	}
}

// Set the lower bound of the uniform distribution to use for state generation
//
// Be aware that rand_min must be strictly less than rand_max to build.
//
// Note a reference to the builder is returned to allow for chaining.
func (builder *StateGeneratorBuilder) SetRandMin(randMin float64) *StateGeneratorBuilder {
	builder.randMin = randMin
	return builder
}

// Set the upper bound of the uniform distribution to use for state generation
//
// Be aware that rand_max must be strictly greater than rand_min to build.
//
// Note a reference to the builder is returned to allow for chaining.
func (builder *StateGeneratorBuilder) SetRandMax(randMax float64) *StateGeneratorBuilder {
	builder.randMax = randMax
	return builder
}

// Set the random seed for the uniform distribution used for state generation.
//
// If the seed is left at the default value (0) then a random seed is created.
//
// Note a reference to the builder is returned to allow for chaining.
func (builder *StateGeneratorBuilder) SetSeed(seed uint64) *StateGeneratorBuilder {
	builder.seed = seed
	return builder
}

// Sets the domain of the state generator
func (builder *StateGeneratorBuilder) SetGeneratorDomain(domain domain.DomainEnum) *StateGeneratorBuilder {
	builder.domain = domain
	return builder
}

// Set the dimension of the vectors to be produced, i.e. the length of the vector.
//
// Dimension must be a strictly positive integer and match the Hopfield Network dimension.
//
// Note a reference to the builder is returned to allow for chaining.
func (builder *StateGeneratorBuilder) SetGeneratorDimension(dimension int) *StateGeneratorBuilder {
	builder.dimension = dimension
	return builder
}

// Perform final checks on the builder - to be run right before constructing the StateGenerator struct.
//
// Note this function panics if builder is invalid - perhaps instead an error could be bubbled up? This is probably okay though.
func (builder *StateGeneratorBuilder) checkValid() {
	if builder.randMin >= builder.randMax {
		panic("StateGeneratorBuilder encountered an error during build! rand_min must be strictly smaller than rand_max!")
	}

	if builder.dimension <= 0 {
		panic("StateGeneratorBuilder encountered an error during build! Dimension must be strictly positive!")
	}
}

// Builds the StateGenerator.
//
// Note this function will panic if invalid options are given to the builder.
//
// RngDistribution must be a valid distribution from the [gonum.org/v1/gonum/stat/distuv] package.
//
// Dimension must be a strictly positive integer and match the Hopfield Network dimension.
func (builder *StateGeneratorBuilder) Build() *StateGenerator {
	builder.checkValid()

	var seed uint64
	if builder.seed == 0 {
		seed = builder.seedGenerator.Uint64()
	} else {
		seed = builder.seed
	}

	rand_dist := distuv.Uniform{
		Min: builder.randMin,
		Max: builder.randMax,
		Src: rand.NewSource(seed),
	}

	return &StateGenerator{
		domainManager: domain.GetDomainManager(builder.domain),
		rng:           rand_dist,
		dimension:     builder.dimension,
	}
}
