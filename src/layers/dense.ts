
/* IMPORT */

import AbstractHidden from '~/layers/abstract_hidden';
import Buffer from '~/buffer';
import Tensor from '~/tensor';
import {range} from '~/utils';
import type Abstract from '~/layers/abstract';
import type {DenseOptions, ParamsAndGrads, Precision} from '~/types';

/* MAIN */

class Dense extends AbstractHidden<DenseOptions> {

  /* VARIABLES */

  bias: number;
  l1decay: number;
  l2decay: number;

  biases: Tensor;
  filters: Tensor[];

  /* CONSTRUCTOR */

  constructor ( options: DenseOptions, prev?: Abstract ) {

    super ( options, prev );

    this.osx = 1;
    this.osy = 1;
    this.osz = options.filters;

    this.bias = options.bias ?? 0;
    this.l1decay = options.l1decay ?? 0;
    this.l2decay = options.l2decay ?? 1;

    this.biases = options._biases ? new Tensor ( 1, 1, this.osz, Buffer.decode ( options._biases ) ) : new Tensor ( 1, 1, this.osz, this.bias );
    this.filters = options._filters ? options._filters.map ( filter => new Tensor ( 1, 1, this.il, Buffer.decode ( filter ) ) ) : range ( 0, this.osz ).map ( () => new Tensor ( 1, 1, this.il ) );

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const output = new Tensor ( 1, 1, this.osz, 0 );

    for ( let i = 0, l = this.osz; i < l; i++ ) {
      let a = 0;
      let wi = this.filters[i].w;
      for ( let d = 0; d < this.il; d++ ) {
        a += input.w[d] * wi[d];
      }
      a += this.biases.w[i];
      output.w[i] = a;
    }

    this.ot = output;

    return this.ot;

  }

  backward (): void {

    const input = this.it;

    input.dw = new Buffer ( input.length );

    for ( let i = 0, l = this.osz; i < l; i++ ) {
      let tfi = this.filters[i];
      let chain_grad = this.ot.dw[i];
      for ( let d = 0; d < this.il; d++ ) {
        input.dw[d] += tfi.w[d] * chain_grad;
        tfi.dw[d] += input.w[d] * chain_grad;
      }
      this.biases.dw[i] += chain_grad;
    }

  }

  getAsOptions ( precision: Precision ): DenseOptions {

    return {
      ...this.options,
      _biases: Buffer.encode ( this.biases.w, precision ),
      _filters: this.filters.map ( filter => Buffer.encode ( filter.w, precision ) )
    };

  }

  getParamsAndGrads (): ParamsAndGrads[] {

    const filters = this.filters.map ( filter => ({ params: filter.w, grads: filter.dw, l1decay: this.l1decay, l2decay: this.l2decay }) );
    const biases = { params: this.biases.w, grads: this.biases.dw, l1decay: 0, l2decay: 0 };

    return [...filters, biases];

  }

}

/* EXPORT */

export default Dense;
