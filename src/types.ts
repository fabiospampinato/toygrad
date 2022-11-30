
/* MAIN */

type Activation = 'relu' | 'sigmoid' | 'softplus' | 'tanh';

type ActivationFN = ( x: number, derivative: boolean ) => number;

type Matrix = number[][];

type Vector = number[];

// type Options = { //TODO: make this nicer
//   inputLayer: number,
//   hiddenLayer: number,
//   outputLayer: number,
//   learningRate: number,
//   epochs: number,
//   activation: Activation
// };

type Layer = {
  neurons: number,
  activation: Activation
};

type Options = { //TODO: make this nicer
  layers: Layer[],
  learningRate: number
};

// log: true,
// logPeriod: TRAIN_LOG_PERIOD

/* EXPORT */

export type {Activation, ActivationFN, Matrix, Vector, Options};
