
/* IMPRT */

import Abstract from '~/layers/abstract';
import AbstractActivation from '~/layers/abstract_activation';
import AbstractHidden from '~/layers/abstract_hidden';
import AbstractInput from '~/layers/abstract_input';
import AbstractOutput from '~/layers/abstract_output';
import Conv from '~/layers/conv';
import Dense from '~/layers/dense';
import Dropout from '~/layers/dropout';
import Input from '~/layers/input';
import LeakyRelu from '~/layers/leakyrelu';
import Maxout from '~/layers/maxout';
import Pool from '~/layers/pool';
import Regression from '~/layers/regression';
import Relu from '~/layers/relu';
import Sigmoid from '~/layers/sigmoid';
import Softmax from '~/layers/softmax';
import Tanh from '~/layers/tanh';

/* MAIN */

const Layers = {
  /* ABSTRACT */
  Abstract,
  AbstractActivation,
  AbstractHidden,
  AbstractInput,
  AbstractOutput,
  /* INPUT */
  Input,
  /* HIDDEN */
  Conv,
  Dense,
  Dropout,
  Pool,
  /* ACTIVATION */
  LeakyRelu,
  Maxout,
  Relu,
  Sigmoid,
  Tanh,
  /* OUTPUT */
  Regression,
  Softmax
};

/* EXPORT */

export default Layers;
