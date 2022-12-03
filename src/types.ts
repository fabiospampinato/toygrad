
/* IMPORT */

import type Matrix from './matrix';

/* MAIN */

type ActivationName = 'leakyrelu' | 'relu' | 'sigmoid' | 'softmax' | 'softplus' | 'tanh';

type ActivationMethodSingle = (( x: number, derivative: boolean ) => number) & { multi?: false };

type ActivationMethodMultiple = (( x: Matrix, derivative: boolean ) => Matrix) & { multi: true };

type ActivationMethod = ActivationMethodSingle | ActivationMethodMultiple;

type Vector = number[];

type Layer = {
  inputs: number,
  outputs: number,
  activation: ActivationName,
  biases?: string,
  weights?: string
};

type Options = {
  layers: Layer[],
  learningRate: number
};

type ResultForward = [
  weighted0: Matrix,
  weighted1: Matrix,
  activated0: Matrix,
  activated1: Matrix
];

type ResultBackward = [
  error0: Matrix,
  error1: Matrix,
  gradient0: Matrix,
  gradient1: Matrix
];

type ResultTrain = [
  inputs: Vector[],
  outputs: Vector[],
  forward: ResultForward,
  backward: ResultBackward
];

/* EXPORT */

export type {ActivationName, ActivationMethod, Vector};
export type {Layer, Options};
export type {ResultForward, ResultBackward, ResultTrain};
