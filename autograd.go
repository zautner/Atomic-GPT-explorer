package main

import "math"

// Value is the core unit in a tiny automatic differentiation engine.
//
// Think of this as a "number with memory":
// - Data is the actual number used in calculations.
// - Grad is "how much the final loss changes if this number changes a little."
// - Children points to the input nodes used to create this value.
// - LocalGrads stores local derivative factors for each child.
//
// This structure allows us to build a computation graph during forward pass
// and then send gradients backward with the chain rule.
type Value struct {
	Data       float64
	Grad       float64
	Children   []*Value
	LocalGrads []float64
}

// NewValue creates a leaf node (a plain number with no parents).
func NewValue(data float64) *Value {
	return &Value{Data: data}
}

// Add creates node z = x + y.
// Local derivatives:
// dz/dx = 1
// dz/dy = 1
func (v *Value) Add(other *Value) *Value {
	return &Value{
		Data:       v.Data + other.Data,
		Children:   []*Value{v, other},
		LocalGrads: []float64{1, 1},
	}
}

// Mul creates node z = x * y.
// Local derivatives:
// dz/dx = y
// dz/dy = x
func (v *Value) Mul(other *Value) *Value {
	return &Value{
		Data:       v.Data * other.Data,
		Children:   []*Value{v, other},
		LocalGrads: []float64{other.Data, v.Data},
	}
}

// Pow creates node z = x^p.
// Local derivative:
// dz/dx = p * x^(p-1)
func (v *Value) Pow(power float64) *Value {
	return &Value{
		Data:       math.Pow(v.Data, power),
		Children:   []*Value{v},
		LocalGrads: []float64{power * math.Pow(v.Data, power-1)},
	}
}

// Log creates node z = ln(x).
// Local derivative:
// dz/dx = 1/x
func (v *Value) Log() *Value {
	return &Value{
		Data:       math.Log(v.Data),
		Children:   []*Value{v},
		LocalGrads: []float64{1 / v.Data},
	}
}

// Exp creates node z = e^x.
// Local derivative:
// dz/dx = e^x
func (v *Value) Exp() *Value {
	exp := math.Exp(v.Data)
	return &Value{
		Data:       exp,
		Children:   []*Value{v},
		LocalGrads: []float64{exp},
	}
}

// Relu applies the ReLU activation:
// relu(x) = max(0, x)
//
// Local derivative:
// 1 when x > 0, otherwise 0.
func (v *Value) Relu() *Value {
	grad := 0.0
	if v.Data > 0 {
		grad = 1.0
	}
	return &Value{
		Data:       math.Max(0, v.Data),
		Children:   []*Value{v},
		LocalGrads: []float64{grad},
	}
}

// Backward performs reverse-mode autodiff from this node to all ancestors.
//
// Process:
// 1) Build topological order so each node is visited only after its children.
// 2) Seed output gradient with 1 (dOutput/dOutput = 1).
// 3) Traverse graph in reverse topological order and accumulate gradients.
func (v *Value) Backward() {
	topo := []*Value{}
	visited := make(map[*Value]bool)

	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.Children {
			buildTopo(child)
		}
		topo = append(topo, node)
	}
	buildTopo(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		curr := topo[i]
		for j, child := range curr.Children {
			child.Grad += curr.LocalGrads[j] * curr.Grad
		}
	}
}
