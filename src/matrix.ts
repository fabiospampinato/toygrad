
/* MAIN */

class Matrix {

  /* VARIABLES */

  rows: number;
  cols: number;
  buffer: Float32Array;

  /* CONSTRUCTOR */

  constructor ( rows: number, cols: number ) {

    this.rows = rows;
    this.cols = cols;
    this.buffer = new Float32Array ( rows * cols );

  }

  /* API */

  get ( row: number, col: number ): number {

    const index = ( row * this.cols ) + col;

    return this.buffer[index];

  }

  set ( row: number, col: number, value: number ): number {

    const index = ( row * this.cols ) + col;

    return this.buffer[index] = value;

  }

}

/* EXPORT */

export default Matrix;
