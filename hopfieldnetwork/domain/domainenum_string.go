// Code generated by "stringer -type DomainEnum"; DO NOT EDIT.

package domain

import "strconv"

func _() {
	// An "invalid array index" compiler error signifies that the constant values have changed.
	// Re-run the stringer command to generate them again.
	var x [1]struct{}
	_ = x[BipolarDomain-0]
	_ = x[BinaryDomain-1]
}

const _DomainEnum_name = "BipolarDomainBinaryDomain"

var _DomainEnum_index = [...]uint8{0, 13, 25}

func (i DomainEnum) String() string {
	if i < 0 || i >= DomainEnum(len(_DomainEnum_index)-1) {
		return "DomainEnum(" + strconv.FormatInt(int64(i), 10) + ")"
	}
	return _DomainEnum_name[_DomainEnum_index[i]:_DomainEnum_index[i+1]]
}
