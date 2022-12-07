
/* IMPORT */

import Abstract from '~/trainers/abstract';
import type NeuralNetwork from '~/neural_network';
import type {TrainerAdadeltaOptions} from '~/types';

/* MAIN */

class Adadelta extends Abstract {

  /* CONSTRUCTOR */

  constructor ( nn: NeuralNetwork, options: TrainerAdadeltaOptions ) {

    super ( nn, {
      method: 'adadelta',
      ...options
    });

  }

}

/* EXPORT */

export default Adadelta;
