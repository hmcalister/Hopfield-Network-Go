package states

import (
	"hmcalister/hopfield/hopfieldnetwork/activationfunction"
	"hmcalister/hopfield/hopfieldnetwork/networkdomain"
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
//
// Domain defines what values can exist within the generated vector.
type StateGeneratorBuilder struct {
	randMin       float64
	randMax       float64
	seed          uint64
	seedGenerator *rand.Rand
	dimension     int
	domain        networkdomain.NetworkDomain
}

func NewStateGeneratorBuilder() *StateGeneratorBuilder {
	seedGenSrc := rand.NewSource(uint64(time.Now().Nanosecond()))
	seedGen := rand.New(seedGenSrc)
	return &StateGeneratorBuilder{
		randMin:       -1,
		randMax:       1,
		seed:          0,
		seedGenerator: seedGen,
		dimension:     0,
		domain:        networkdomain.UnspecifiedDomain,
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

// Set the dimension of the vectors to be produced, i.e. the length of the vector.
//
// Dimension must be a strictly positive integer and match the Hopfield Network dimension.
//
// Note a reference to the builder is returned to allow for chaining.
func (builder *StateGeneratorBuilder) SetGeneratorDimension(dimension int) *StateGeneratorBuilder {
	builder.dimension = dimension
	return builder
}

// Set the domain of the StateGenerator. This will in turn set the activation function to be used
// to ensure states end up as valid.
//
// Domain must be a valid networkDomain from the [hmcalister/hopfield/hopfieldnetwork/networkdomain] subpackage.
//
// Note a reference to the builder is returned to allow for chaining.
func (builder *StateGeneratorBuilder) SetGeneratorDomain(domain networkdomain.NetworkDomain) *StateGeneratorBuilder {
	builder.domain = domain
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

	if builder.domain == networkdomain.UnspecifiedDomain {
		panic("StateGeneratorBuilder encountered an error during build! Domain must be a valid network domain!")
	}
}

// Builds the StateGenerator.
//
// Note this function will panic if invalid options are given to the builder.
//
// RngDistribution must be a valid distribution from the [gonum.org/v1/gonum/stat/distuv] package.
//
// Dimension must be a strictly positive integer and match the Hopfield Network dimension.
//
// Domain must be a valid networkDomain from the [hmcalister/hopfield/hopfieldnetwork/networkdomain] subpackage.
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

	activationFunction := activationfunction.GetDomainActivationFunction(builder.domain)

	return &StateGenerator{
		rng:                rand_dist,
		dimension:          builder.dimension,
		activationFunction: activationFunction,
	}
}
