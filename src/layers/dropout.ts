
/* IMPORT */

import Buffer from '~/buffer';
import AbstractHidden from '~/layers/abstract_hidden';
import type Abstract from '~/layers/abstract';
import type Tensor from '~/tensor';
import type {DropoutOptions} from '~/types';

/* MAIN */

class Dropout extends AbstractHidden<DropoutOptions> {

  /* VARIABLES */

  probability: number;
  dropped: Float32Array;

  /* CONSTRUCTOR */

  constructor ( options: DropoutOptions, prev?: Abstract ) {

    super ( options, prev );

    this.probability = options.probability ?? 0.5;
    this.dropped = new Buffer ( this.osx * this.osy * this.osz );

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const output = input.clone ();

    if ( isTraining ) {
      // do dropout
      for ( let i = 0; i < input.length; i++ ) {
        if ( Math.random () < this.probability ) {  // drop!
          output.w[i] = 0;
          this.dropped[i] = 1;
        } else {
          this.dropped[i] = 0;
        }
      }
    } else {
      // scale the activations during prediction
      for ( let i = 0; i < input.length; i++) {
        output.w[i] *= this.probability;
      }
    }

    this.ot = output;

    return this.ot; // dummy identity function for now

  }

  backward (): void {

    const input = this.it;
    const output = this.ot;

    input.dw = new Buffer ( input.length );
    for ( let i = 0, l = input.length; i < l; i++ ) {
      if ( !this.dropped[i] ) {
        input.dw[i] = output.dw[i];
      }
    }

  }

}

/* EXPORT */

export default Dropout;
