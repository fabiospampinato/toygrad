
/* IMPORT */

import Buffer from '~/buffer';
import AbstractHidden from '~/layers/abstract_hidden';
import Tensor from '~/tensor';
import {range} from '~/utils';
import type Abstract from '~/layers/abstract';
import type {ConvOptions, ParamsAndGrads, Precision} from '~/types';

/* MAIN */

class Conv extends AbstractHidden<ConvOptions> {

  /* VARIABLES */

  sx: number;
  sy: number;
  stride: number;
  pad: number;

  bias: number;
  l1decay: number;
  l2decay: number;

  biases: Tensor;
  filters: Tensor[];

  /* CONSTRUCTOR */

  constructor ( options: ConvOptions, prev?: Abstract ) {

    super ( options, prev );

    this.sx = options.sx;
    this.sy = options.sy ?? this.sx;
    this.stride = options.stride;
    this.pad = options.pad;

    this.osx = Math.floor ( ( ( this.isx + ( this.pad * 2 ) - this.sx ) / this.stride ) + 1 );
    this.osy = Math.floor ( ( ( this.isy + ( this.pad * 2 ) - this.sy ) / this.stride ) + 1 );
    this.osz = options.filters;

    this.bias = options.bias ?? 0;
    this.l1decay = options.l1decay ?? 0;
    this.l2decay = options.l2decay ?? 1;

    this.biases = options._biases ? new Tensor ( 1, 1, this.osz, Buffer.decode ( options._biases ) ) : new Tensor ( 1, 1, this.osz, this.bias );
    this.filters = options._filters ? options._filters.map ( filter => new Tensor ( this.sx, this.sy, this.isz, Buffer.decode ( filter ) ) ) : range ( 0, this.osz ).map ( () => new Tensor ( this.sx, this.sy, this.isz ) );

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    this.it = input;

    const output = new Tensor ( this.osx, this.osy, this.osz, 0 );

    let V_sx = input.sx;
    let V_sy = input.sy;
    let xy_stride = this.stride;

    for(let d=0;d<this.osz;d++) {
      let f = this.filters[d];
      let x = -this.pad;
      let y = -this.pad;
      for(let ay=0; ay<this.osy; y+=xy_stride,ay++) {  // xy_stride
        x = -this.pad;
        for(let ax=0; ax<this.osx; x+=xy_stride,ax++) {  // xy_stride
          // convolve centered at this particular location
          let a = 0;
          for(let fy=0;fy<f.sy;fy++) {
            let oy = y+fy; // coordinates in the original input array coordinates
            for(let fx=0;fx<f.sx;fx++) {
              let ox = x+fx;
              if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                for(let fd=0;fd<f.sz;fd++) {
                  // avoid function call overhead (x2) for efficiency, compromise modularity :(
                  a += f.w[((f.sx * fy)+fx)*f.sz+fd] * input.w[((V_sx * oy)+ox)*input.sz+fd];
                }
              }
            }
          }
          a += this.biases.w[d];
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

    let V_sx = input.sx;
    let V_sy = input.sy;
    let xy_stride = this.stride;

    for(let d=0;d<this.osz;d++) {
      let f = this.filters[d];
      let x = -this.pad;
      let y = -this.pad;
      for(let ay=0; ay<this.osy; y+=xy_stride,ay++) {  // xy_stride
        x = -this.pad;
        for(let ax=0; ax<this.osx; x+=xy_stride,ax++) {  // xy_stride
          // convolve centered at this particular location
          let chain_grad = this.ot.getGrad(ax,ay,d); // gradient from above, from chain rule
          for(let fy=0;fy<f.sy;fy++) {
            let oy = y+fy; // coordinates in the original input array coordinates
            for(let fx=0;fx<f.sx;fx++) {
              let ox = x+fx;
              if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                for(let fd=0;fd<f.sz;fd++) {
                  // avoid function call overhead (x2) for efficiency, compromise modularity :(
                  let ix1 = ((V_sx * oy)+ox)*input.sz+fd;
                  let ix2 = ((f.sx * fy)+fx)*f.sz+fd;
                  f.dw[ix2] += input.w[ix1]*chain_grad;
                  input.dw[ix1] += f.w[ix2]*chain_grad;
                }
              }
            }
          }
          this.biases.dw[d] += chain_grad;
        }
      }
    }

  }

  getAsOptions ( precision: Precision ): ConvOptions {

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

export default Conv;
