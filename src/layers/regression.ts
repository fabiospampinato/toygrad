
/* IMPORT */

import Buffer from '~/buffer';
import AbstractOutput from '~/layers/abstract_output';
import type Abstract from '~/layers/abstract';
import type Tensor from '~/tensor';
import type {RegressionOptions} from '~/types';

/* MAIN */

class Regression extends AbstractOutput<RegressionOptions> {

  /* CONSTRUCTOR */

  constructor ( options: RegressionOptions, prev?: Abstract ) {

    super ( options, prev );

    this.osx = 1;
    this.osy = 1;
    this.osz = this.il;

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;
    this.ot = input;

    return input;

  }

  backward ( output: ArrayLike<number> ): number {

    const input = this.it;

    input.dw = new Buffer ( input.length );

    let loss = 0;

    for ( let i = 0, l = this.osz; i < l; i++ ) {
      let dy = input.w[i] - output[i];
      input.dw[i] = dy;
      loss += 0.5 * dy * dy;
    }

    return loss;

  }

}

/* EXPORT */

export default Regression;
