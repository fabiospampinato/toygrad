
/* IMPORT */

import Buffer from '~/buffer';
import AbstractOutput from '~/layers/abstract_output';
import Tensor from '~/tensor';
import type Abstract from '~/layers/abstract';
import type {SoftmaxOptions} from '~/types';

/* MAIN */

class Softmax extends AbstractOutput<SoftmaxOptions> {

  /* VARIABLES */

  es!: Buffer;

  /* CONSTRUCTOR */

  constructor ( options: SoftmaxOptions, prev?: Abstract ) {

    super ( options, prev );

    this.osx = 1;
    this.osy = 1;
    this.osz = this.il;

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const output = new Tensor ( 1, 1, this.osz, 0 );

    // compute max activation
    let as = input.w;
    let amax = input.w[0];
    for ( let i = 1, l = this.osz; i < l; i++ ) {
      if ( as[i] > amax ) amax = as[i];
    }

    // compute exponentials (carefully to not blow up)
    let es = new Buffer ( this.osz );
    let esum = 0;
    for ( let i = 0, l = this.osz; i < l; i++ ) {
      let e = Math.exp ( as[i] - amax );
      esum += e;
      es[i] = e;
    }

    // normalize and output to sum to one
    for ( let i = 0, l = this.osz; i < l; i++ ) {
      es[i] /= esum;
      output.w[i] = es[i];
    }

    this.es = es;
    this.ot = output;

    return this.ot;

  }

  backward ( output: number ): number {

    const input = this.it;

    input.dw = new Buffer ( input.length );

    for ( let i = 0, l = this.osz; i < l; i++ ) {
      const indicator = i === output ? 1 : 0;
      const mul = -( indicator - this.es[i] );
      input.dw[i] = mul;
    }

    // loss is the class negative log likelihood
    return - Math.log ( this.es[output] );

  }

}

/* EXPORT */

export default Softmax;
