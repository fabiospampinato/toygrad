
/* IMPORT */

import type Tensor from '~/tensor';
import type {AbstractOptions, ParamsAndGrads, Precision} from '~/types';

/* MAIN */

class Abstract<T extends AbstractOptions = AbstractOptions> {

  /* VARIABLES */

  options: T;

  isx: number;
  isy: number;
  isz: number;
  il: number;
  it!: Tensor;

  osx: number;
  osy: number;
  osz: number;
  ot!: Tensor;

  /* CONSTRUCTOR */

  constructor ( options: T, prev?: Abstract ) {

    this.options = options;

    this.isx = prev?.osx ?? -1;
    this.isy = prev?.osy ?? -1;
    this.isz = prev?.osz ?? -1;
    this.il = this.isx * this.isy * this.isz;

    this.osx = this.isx;
    this.osy = this.isy;
    this.osz = this.isz;

  }

  /* API */

  forward ( input: Tensor, isTraining: boolean ): Tensor {

    throw new Error ( 'Not implemented' );

  }

  backward ( output: any ): any {

    throw new Error ( 'Not implemented' );

  }

  getAsOptions ( precision: Precision ): T {

    return this.options;

  }

  getParamsAndGrads (): ParamsAndGrads[] {

    return [];

  }

}

/* EXPORT */

export default Abstract;
