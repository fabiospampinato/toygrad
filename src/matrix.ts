
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

  /* STATIC API */

  static from ( source: Matrix | number[][] ): Matrix {

    if ( !Array.isArray ( source ) ) return source; // Fast-path, pre-allocated Matrix

    const rows = source.length;
    const cols = source[0].length;
    const matrix = new Matrix ( rows, cols );

    for ( let i = 0; i < rows; i++ ) {
      for ( let j = 0; j < cols; j++ ) {
        matrix.set ( i, j, source[i][j] );
      }
    }

    return matrix;

  }

}

/* EXPORT */

export default Matrix;
