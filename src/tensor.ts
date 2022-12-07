
/* IMPORT */

import Buffer from '~/buffer';
import {isBuffer, isNumber, isUndefined, randn} from '~/utils';

/* MAIN */

//TODO: Generalize this to an arbitrary number of dimensions

class Tensor {

  /* VARIABLES */

  sx: number;
  sy: number;
  sz: number;
  length: number;
  w: Float32Array;
  dw: Float32Array;

  /* CONSTRUCTOR */

  constructor ( sx: number, sy: number, sz: number, value?: Float32Array | ArrayLike<number> | number ) {

    this.sx = sx;
    this.sy = sy;
    this.sz = sz;
    this.length = sx * sy * sz;

    /* INITIALIZING WEIGHTS */

    if ( isUndefined ( value ) ) { // With normalized random values

      const scale = Math.sqrt ( 1 / this.length );
      const get = () => randn ( 0, scale );

      this.w = new Buffer ( this.length ).map ( get );
      this.dw = new Buffer ( this.length );

    } else if ( isNumber ( value ) ) { // With a fixed value

      this.w = new Buffer ( this.length );
      this.dw = new Buffer ( this.length );

      if ( value !== 0 ) {
        this.w.fill ( value );
      }

    } else if ( isBuffer ( value ) ) { // With an existing buffer

      this.w = value;
      this.dw = new Buffer ( this.length );

    } else { // With an existing array

      this.w = new Buffer ( value );
      this.dw = new Buffer ( this.length );

    }

  }

  /* WEIGHTS API */

  index ( x: number, y: number, z: number ): number {

    return ( ( ( this.sx * y ) + x ) * this.sz ) + z;

  }

  get ( x: number, y: number, z: number ): number {

    return this.w[this.index ( x, y, z )];

  }

  set ( x: number, y: number, z: number, value: number ): number {

    return this.w[this.index ( x, y, z )] = value;

  }

  add ( x: number, y: number, z: number, value: number ): number {

    return this.w[this.index ( x, y, z )] += value;

  }

  /* GRADIENT API */

  getGrad ( x: number, y: number, z: number ): number {

    return this.dw[this.index ( x, y, z )];

  }

  setGrad ( x: number, y: number, z: number, value: number ): number {

    return this.dw[this.index ( x, y, z )] = value;

  }

  addGrad ( x: number, y: number, z: number, value: number ): number {

    return this.dw[this.index ( x, y, z )] += value;

  }

  /* CLONE API */

  clone (): Tensor {

    const clone = new Tensor ( this.sx, this.sy, this.sz, this.w );

    clone.w = clone.w.slice ();

    return clone;

  }

  cloneWithZeros (): Tensor {

    return new Tensor ( this.sx, this.sy, this.sz, 0 );

  }

}

/* EXPORT */

export default Tensor;
