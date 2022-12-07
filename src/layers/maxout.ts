
/* IMPORT */

import Buffer from '~/buffer';
import AbstractHidden from '~/layers/abstract_hidden';
import Tensor from '~/tensor';
import type Abstract from '~/layers/abstract';
import type {MaxoutOptions} from '~/types';

/* MAIN */

class Maxout extends AbstractHidden {

  /* VARIABLES */

  sx: number;
  switches: Float32Array;

  /* CONSTRUCTOR */

  constructor ( options: MaxoutOptions, prev?: Abstract ) {

    super ( options, prev );

    this.sx = options.sx ?? 2;
    this.osz = Math.floor ( this.isz / this.sx );
    this.switches = new Buffer ( this.osx * this.osy * this.osz );

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const N = this.osz;
    const output = new Tensor ( this.osx, this.osy, this.osz, 0 );

    // optimization branch. If we're operating on 1D arrays we dont have
    // to worry about keeping track of x,y,d coordinates inside
    // input volumes. In convnets we do :(
    if(this.osx === 1 && this.osy === 1) {
      for(let i=0;i<N;i++) {
        let ix = i * this.sx; // base index offset
        let a = input.w[ix];
        let ai = 0;
        for(let j=1;j<this.sx;j++) {
          let a2 = input.w[ix+j];
          if(a2 > a) {
            a = a2;
            ai = j;
          }
        }
        output.w[i] = a;
        this.switches[i] = ix + ai;
      }
    } else {
      let n=0; // counter for switches
      for(let x=0;x<input.sx;x++) {
        for(let y=0;y<input.sy;y++) {
          for(let i=0;i<N;i++) {
            let ix = i * this.sx;
            let a = input.get(x, y, ix);
            let ai = 0;
            for(let j=1;j<this.sx;j++) {
              let a2 = input.get(x, y, ix+j);
              if(a2 > a) {
                a = a2;
                ai = j;
              }
            }
            output.set(x,y,i,a);
            this.switches[n] = ix + ai;
            n++;
          }
        }
      }

    }

    this.ot = output;

    return this.ot;

  }

  backward (): void {

    const input = this.it;
    const output = this.ot;

    input.dw = new Buffer ( input.length );

    // pass the gradient through the appropriate switch
    if(this.osx === 1 && this.osy === 1) {
      for(let i=0;i<this.osz;i++) {
        let chain_grad = output.dw[i];
        input.dw[this.switches[i]] = chain_grad;
      }
    } else {
      // bleh okay, lets do this the hard way
      let n=0; // counter for switches
      for(let x=0;x<output.sx;x++) {
        for(let y=0;y<output.sy;y++) {
          for(let i=0;i<this.osz;i++) {
            let chain_grad = output.getGrad(x,y,i);
            input.setGrad(x,y,this.switches[n],chain_grad);
            n++;
          }
        }
      }
    }

  }

}

/* EXPORT */

export default Maxout;
