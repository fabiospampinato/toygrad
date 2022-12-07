
/* IMPORT */

import Buffer from '~/buffer';
import AbstractActivation from '~/layers/abstract_activation';
import type Tensor from '~/tensor';
import type {ActivationElementwiseOptions} from '~/types';

/* MAIN */

class AbstractActivationElementwise<T extends ActivationElementwiseOptions> extends AbstractActivation<T> {

  /* API */

  activationForward ( x: number ): number {

    throw new Error ( 'Unimplemented' );

  }

  activationBackward ( x: number, dx: number ): number {

    throw new Error ( 'Unimplemented' );

  }

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const output = input.clone ();

    for ( let i = 0, l = input.length; i < l; i++ ) {

      output.w[i] = this.activationForward ( output.w[i] );

    }

    this.ot = output;

    return this.ot;

  }

  backward (): void {

    const input = this.it;
    const output = this.ot;

    input.dw = new Buffer ( input.length );

    for ( let i = 0, l = input.length; i < l; i++ ) {

      input.dw[i] = this.activationBackward ( output.w[i], output.dw[i] );

    }

  }

}

/* EXPORT */

export default AbstractActivationElementwise;
