
/* IMPORT */

import type Matrix from './matrix';

/* MAIN */

type Activation = 'leakyrelu' | 'relu' | 'sigmoid' | 'softplus' | 'tanh';

type ActivationFN = ( x: number, derivative: boolean ) => number;

type Vector = number[];

type Layer = {
  inputs: number,
  outputs: number,
  activation: Activation,
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

export type {Activation, ActivationFN, Vector};
export type {Layer, Options};
export type {ResultForward, ResultBackward, ResultTrain};
