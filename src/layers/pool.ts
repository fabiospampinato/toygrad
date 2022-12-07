
/* IMPORT */

import Buffer from '~/buffer';
import AbstractHidden from '~/layers/abstract_hidden';
import Tensor from '~/tensor';
import type Abstract from '~/layers/abstract';
import type {PoolOptions} from '~/types';

/* MAIN */

class Pool extends AbstractHidden<PoolOptions> {

  /* VARIABLES */

  sx: number;
  sy: number;
  stride: number;
  pad: number;

  switchX: Float32Array;
  switchY: Float32Array;

  /* CONSTRUCTOR */

  constructor ( options: PoolOptions, prev?: Abstract ) {

    super ( options, prev );

    this.sx = options.sx;
    this.sy = options.sy ?? this.sx;
    this.stride = options.stride ?? 2;
    this.pad = options.pad ?? 0;

    this.osx = Math.floor ( ( ( this.isx + ( this.pad * 2 ) - this.sx ) / this.stride ) + 1 );
    this.osy = Math.floor ( ( ( this.isy + ( this.pad * 2 ) - this.sy ) / this.stride ) + 1 );
    this.osz = this.isz;

    this.switchX = new Buffer ( this.osx * this.osy * this.osz );
    this.switchY = new Buffer ( this.osx * this.osy * this.osz );

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const output = new Tensor ( this.osx, this.osy, this.osz, 0 );

    let n=0; // a counter for switches
    for(let d=0;d<this.osz;d++) {
      let x = -this.pad;
      let y = -this.pad;
      for(let ax=0; ax<this.osx; x+=this.stride,ax++) {
        y = -this.pad;
        for(let ay=0; ay<this.osy; y+=this.stride,ay++) {
          // convolve centered at this particular location
          let a = -99999; // hopefully small enough ;\
          let winx=-1,winy=-1;
          for(let fx=0;fx<this.sx;fx++) {
            for(let fy=0;fy<this.sy;fy++) {
              let oy = y+fy;
              let ox = x+fx;
              if(oy>=0 && oy<input.sy && ox>=0 && ox<input.sx) {
                let v = input.get(ox, oy, d);
                // perform max pooling and store pointers to where
                // the max came from. This will speed up backprop
                // and can help make nice visualizations in future
                if(v > a) { a = v; winx=ox; winy=oy;}
              }
            }
          }
          this.switchX[n] = winx;
          this.switchY[n] = winy;
          n++;
          output.set(ax, ay, d, a);
        }
      }
    }

    this.ot = output;

    return this.ot;

  }

  backward (): void {

    const input = this.it;

    input.dw = new Buffer ( input.length );

    let n = 0;
    for(let d=0;d<this.osz;d++) {
      let x = -this.pad;
      let y = -this.pad;
      for(let ax=0; ax<this.osx; x+=this.stride,ax++) {
        y = -this.pad;
        for(let ay=0; ay<this.osy; y+=this.stride,ay++) {
          let chain_grad = this.ot.getGrad(ax,ay,d);
          input.addGrad(this.switchX[n], this.switchY[n], d, chain_grad);
          n++;
        }
      }
    }

  }

}

/* EXPORT */

export default Pool;
