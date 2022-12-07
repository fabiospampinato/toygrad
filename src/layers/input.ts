
/* IMPORT */

import AbstractInput from '~/layers/abstract_input';
import type Abstract from '~/layers/abstract';
import type Tensor from '~/tensor';
import type {InputOptions} from '~/types';

/* MAIN */

class Input extends AbstractInput<InputOptions> {

  /* CONSTRUCTOR */

  constructor ( options: InputOptions, prev?: Abstract ) {

    super ( options, prev );

    this.osx = options.sx;
    this.osy = options.sy;
    this.osz = options.sz;

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;
    this.ot = input;

    return this.ot;

  }

}

/* EXPORT */

export default Input;
