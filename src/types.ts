
/* LAYERS - ABSTRACT */

type AbstractOptions = {};

type AbstractActivationOptions = {};

type ActivationElementwiseOptions = {};

type AbstractHiddenOptions = {};

type AbstractInputOptions = {};

type AbstractOutputOptions = {};

/* LAYERS - INPUT */

type InputOptions = {
  sx: number,
  sy: number,
  sz: number
};

/* LAYERS - HIDDEN */

type ConvOptions = {
  sx: number,
  sy?: number,
  filters: number,
  stride: number,
  pad: number,
  bias?: number,
  l1decay?: number,
  l2decay?: number,
  _biases?: string,
  _filters?: string[]
};

type DenseOptions = {
  filters: number,
  bias?: number,
  l1decay?: number,
  l2decay?: number,
  _biases?: string,
  _filters?: string[]
};

type DropoutOptions = {
  probability?: number
};

type PoolOptions = {
  sx: number,
  sy?: number,
  stride: number,
  pad?: number
};

/* LAYERS - ACTIVATION */

type LeakyReluOptions = {};

type MaxoutOptions = {
  sx?: number
};

type ReluOptions = {};

type SigmoidOptions = {};

type TanhOptions = {};

/* LAYERS - OUTPUT */

type RegressionOptions = {};

type SoftmaxOptions = {};

/* LAYERS - CLASSES */

type LayerAbstractActivation = import ( '~/layers/abstract_activation' ).default;
type LayerAbstractHidden = import ( '~/layers/abstract_hidden' ).default;
type LayerAbstractInput = import ( '~/layers/abstract_input' ).default;
type LayerAbstractOutput = import ( '~/layers/abstract_output' ).default;
type LayerInput = import ( '~/layers/input' ).default;
type LayerConv = import ( '~/layers/conv' ).default;
type LayerDense = import ( '~/layers/dense' ).default;
type LayerDropout = import ( '~/layers/dropout' ).default;
type LayerPool = import ( '~/layers/pool' ).default;
type LayerLeakyRelu = import ( '~/layers/leakyrelu' ).default;
type LayerMaxout = import ( '~/layers/maxout' ).default;
type LayerRelu = import ( '~/layers/relu' ).default;
type LayerSigmoid = import ( '~/layers/sigmoid' ).default;
type LayerTanh = import ( '~/layers/tanh' ).default;
type LayerRegression = import ( '~/layers/regression' ).default;
type LayerSoftmax = import ( '~/layers/softmax' ).default;

type LayersAbstract = LayerAbstractActivation | LayerAbstractHidden | LayerAbstractInput | LayerAbstractOutput;
type LayersInput = LayerInput;
type LayersHidden = LayerConv | LayerDense | LayerDropout | LayerPool | LayersActivation;
type LayersActivation = LayerLeakyRelu | LayerMaxout | LayerRelu | LayerSigmoid | LayerTanh;
type LayersOutput = LayerRegression | LayerSoftmax;
type Layers = LayersAbstract | LayersInput | LayersHidden | LayersOutput;

/* TRAINERS */

type TrainerAbstractSharedOptions = {
  batchSize?: number,
  learningRate?: number,
  l1decay?: number,
  l2decay?: number,
  momentum?: number
};

type TrainerAbstractSpecificOptions = {
  ro?: number,
  eps?: number,
  beta1?: number,
  beta2?: number
};

type TrainerAbstractOptions = TrainerAbstractSharedOptions & TrainerAbstractSpecificOptions & {
  method?: 'sgd' | 'adam' | 'adagrad' | 'adadelta' | 'nesterov',
};

type TrainerAdamOptions = TrainerAbstractSharedOptions & {
  eps?: number,
  beta1?: number,
  beta2?: number
};

type TrainerAdagradOptions = TrainerAbstractSharedOptions & {
  eps?: number
};

type TrainerAdadeltaOptions = TrainerAbstractSharedOptions & {
  ro?: number,
  eps?: number
};

type TrainerNesterovOptions = TrainerAbstractSharedOptions & {};

type TrainerSGDOptions = TrainerAbstractSharedOptions & {};

type TrainerResult = {
  cost: number,
  loss: number,
  l1loss: number,
  l2loss: number
};

/* NEURAL NETWORK */

type NeuralNetworkLayerInput = { type: 'input' } & InputOptions;
type NeuralNetworkLayerConv = { type: 'conv' } & ConvOptions;
type NeuralNetworkLayerDense = { type: 'dense' } & DenseOptions;
type NeuralNetworkLayerDropout = { type: 'dropout' } & DropoutOptions;
type NeuralNetworkLayerPool = { type: 'pool' } & PoolOptions;
type NeuralNetworkLayerLeakyRelu = { type: 'leakyrelu' } & LeakyReluOptions;
type NeuralNetworkLayerMaxout = { type: 'maxout' } & MaxoutOptions;
type NeuralNetworkLayerRelu = { type: 'relu' } & ReluOptions;
type NeuralNetworkLayerSigmoid = { type: 'sigmoid' } & SigmoidOptions;
type NeuralNetworkLayerTanh = { type: 'tanh' } & TanhOptions;
type NeuralNetworkLayerRegression = { type: 'regression' } & RegressionOptions;
type NeuralNetworkLayerSoftmax = { type: 'softmax' } & SoftmaxOptions;

type NeuralNetworkLayersInput = NeuralNetworkLayerInput;
type NeuralNetworkLayersHidden = NeuralNetworkLayerConv | NeuralNetworkLayerDense | NeuralNetworkLayerDropout | NeuralNetworkLayerPool | NeuralNetworkLayersActivation;
type NeuralNetworkLayersActivation = NeuralNetworkLayerLeakyRelu | NeuralNetworkLayerMaxout | NeuralNetworkLayerRelu | NeuralNetworkLayerSigmoid | NeuralNetworkLayerTanh;
type NeuralNetworkLayersOutput = NeuralNetworkLayerRegression | NeuralNetworkLayerSoftmax;
type NeuralNetworkLayers = NeuralNetworkLayersInput | NeuralNetworkLayersHidden | NeuralNetworkLayersOutput;

type NeuralNetworkLayersDescriptions = [
  NeuralNetworkLayersInput | LayerAbstractInput,
  ...(NeuralNetworkLayersHidden | LayerAbstractHidden)[],
  NeuralNetworkLayersOutput | LayerAbstractOutput
];

type NeuralNetworkLayersResolved = [
  LayersInput | LayerAbstractInput,
  ...(LayersHidden | LayerAbstractHidden)[],
  LayersOutput | LayerAbstractOutput
];

type NeuralNetworkOptions = {
  layers: NeuralNetworkLayersDescriptions
};

/* OTHERS */

type ParamsAndGrads = {
  params: Float32Array,
  grads: Float32Array,
  l1decay: number,
  l2decay: number
};

type Precision = 'f6' | 'f8' | 'f16' | 'f32';

/* EXPORT */

export type {AbstractOptions, AbstractActivationOptions, ActivationElementwiseOptions, AbstractHiddenOptions, AbstractInputOptions, AbstractOutputOptions};
export type {InputOptions};
export type {ConvOptions, DenseOptions, DropoutOptions, PoolOptions};
export type {LeakyReluOptions, MaxoutOptions, ReluOptions, SigmoidOptions, TanhOptions};
export type {RegressionOptions, SoftmaxOptions};
export type {LayersAbstract, LayersInput, LayersHidden, LayersActivation, LayersOutput, Layers};
export type {TrainerAbstractOptions, TrainerAdamOptions, TrainerAdagradOptions, TrainerAdadeltaOptions, TrainerNesterovOptions, TrainerSGDOptions, TrainerResult};
export type {NeuralNetworkLayers, NeuralNetworkLayersDescriptions, NeuralNetworkLayersResolved, NeuralNetworkOptions};
export type {ParamsAndGrads, Precision};
