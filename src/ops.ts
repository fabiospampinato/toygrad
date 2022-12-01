
/* IMPORT */

import type {Matrix} from './types';

/* MAIN */

// These functions are pure, arguments are never mutated and the output depends solely on the input

const abs = ( x: Matrix ): Matrix => {
  return map ( x, Math.abs );
};

const add = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x + y );
};

const clone = ( x: Matrix ): Matrix => {
  return map ( x, x => x );
};

const create = ( rows: number, cols: number, value: number = 0 ): Matrix => {
  const matrix: Matrix = new Array ( rows );
  for ( let i = 0; i < rows; i++ ) {
    matrix[i] = new Array ( cols ).fill ( value );
  }
  return matrix;
};

const divide = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x / y );
};

const each = ( x: Matrix, iterator: ( x: number, row: number, col: number ) => void ): void => {
  for ( let i = 0, l = x.length; i < l; i++ ) {
    for ( let j = 0, m = x[i].length; j < m; j++ ) {
      iterator ( x[i][j], i, j );
    }
  }
};

const each2 = ( x: Matrix, y: Matrix, iterator: ( x: number, y: number, row: number, col: number ) => number ): void => {
  for ( let i = 0, l = x.length; i < l; i++ ) {
    for ( let j = 0, m = x[i].length; j < m; j++ ) {
      iterator ( x[i][j], y[i][j], i, j );
    }
  }
};

const map = ( x: Matrix, iterator: ( x: number, row: number, col: number ) => number ): Matrix => {
  const matrix: Matrix = new Array ( x.length );
  for ( let i = 0, l = x.length; i < l; i++ ) {
    matrix[i] = new Array ( x[i].length );
    for ( let j = 0, m = x[i].length; j < m; j++ ) {
      matrix[i][j] = iterator ( x[i][j], i, j );
    }
  }
  return matrix;
};

const map2 = ( x: Matrix, y: Matrix, iterator: ( x: number, y: number, row: number, col: number ) => number ): Matrix => {
  const matrix: Matrix = new Array ( x.length );
  for ( let i = 0, l = x.length; i < l; i++ ) {
    matrix[i] = new Array ( x[i].length );
    for ( let j = 0, m = x[i].length; j < m; j++ ) {
      matrix[i][j] = iterator ( x[i][j], y[i][j], i, j );
    }
  }
  return matrix;
};

const mean = ( x: Matrix ): number => {
  if ( !x.length ) return 0;
  return sum ( x ) / ( x.length * x[0].length );
};

const multiply = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x * y );
};

const product = ( x: Matrix, y: Matrix ): Matrix => {
  const matrix: Matrix = new Array ( x.length );
  for ( let i = 0, l = x.length; i < l; i++ ) {
    matrix[i] = new Array ( y[0].length );
    for ( let j = 0, m = y[0].length; j < m; j++) {
      let sum = 0;
      for ( let k = 0, n = x[0].length; k < n; k++ ) {
        sum += x[i][k] * y[k][j];
      }
      matrix[i][j] = sum;
    }
  }
  return matrix;
};

const random = ( rows: number, cols: number, min: number, max: number ): Matrix => {
  const rand = () => ( Math.random () * ( max - min ) ) + min;
  return map ( create ( rows, cols ), rand );
};

const reduce = <T> ( x: Matrix, iterator: ( acc: T, x: number, row: number, col: number ) => T, acc: T ): T => {
  each ( x, ( x, row, col ) => {
    acc = iterator ( acc, x, row, col );
  });
  return acc;
};

const scale = ( x: Matrix, factor: number ): Matrix => {
  if ( factor === 1 ) return x;
  return map ( x, x => x * factor );
};

const subtract = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x - y );
};

const sum = ( x: Matrix ): number => {
  return reduce ( x, ( acc, x ) => acc + x, 0 );
};

const transpose = ( x: Matrix ): Matrix => {
  if ( !x.length ) return [];
  const matrix: Matrix = new Array ( x[0].length );
  for ( let i = 0, l = x[0].length; i < l; i++ ) {
    matrix[i] = new Array ( x.length );
    for ( let j = 0, m = x.length; j < m; j++ ) {
      matrix[i][j] = x[j][i];
    }
  }
  return matrix;
};

/* EXPORT */

export {abs, add, clone, create, divide, each, each2, map, map2, mean, multiply, product, random, reduce, scale, subtract, sum, transpose};
