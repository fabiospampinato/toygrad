
/* IMPORT */

import Matrix from './matrix';
import type {Identity} from './types';

/* MAIN */

// These functions are pure, arguments are never mutated and the output depends solely on the input

const abs = ( x: Matrix ): Matrix => {
  return map ( x, Math.abs );
};

const add = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x + y );
};

const divide = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x / y );
};

const each = ( x: Matrix, iterator: ( x: number, row: number, col: number ) => void ): void => {
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      iterator ( x.get ( i, j ), i, j );
    }
  }
};

const each2 = ( x: Matrix, y: Matrix, iterator: ( x: number, y: number, row: number, col: number ) => number ): void => {
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      iterator ( x.get ( i, j ), y.get ( i, j ), i, j );
    }
  }
};

const map = ( x: Matrix, iterator: ( x: number, row: number, col: number ) => number ): Matrix => {
  const matrix = new Matrix ( x.rows, x.cols );
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      matrix.set ( i, j, iterator ( x.get ( i, j ), i, j ) );
    }
  }
  return matrix;
};

const map2 = ( x: Matrix, y: Matrix, iterator: ( x: number, y: number, row: number, col: number ) => number ): Matrix => {
  const matrix = new Matrix ( x.rows, x.cols );
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      matrix.set ( i, j, iterator ( x.get ( i, j ), y.get ( i, j ), i, j ) );
    }
  }
  return matrix;
};

const mean = ( x: Matrix ): number => {
  if ( !x.rows || !x.cols ) return 0;
  return sum ( x ) / ( x.rows * x.cols );
};

const multiply = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x * y );
};

const product = ( x: Matrix, y: Matrix ): Matrix => {
  const matrix = new Matrix ( x.rows, y.cols );
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = y.cols; j < m; j++ ) {
      let sum = 0;
      for ( let k = 0, n = x.cols; k < n; k++ ) {
        sum += x.get ( i, k ) * y.get ( k, j );
      }
      matrix.set ( i, j, sum );
    }
  }
  return matrix;
};

const random = ( rows: number, cols: number, min: number, max: number ): Matrix => {
  const matrix = new Matrix ( rows, cols );
  const rand = () => ( Math.random () * ( max - min ) ) + min;
  return map ( matrix, rand );
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
  if ( !x.rows || !x.cols ) return x;
  const matrix = new Matrix ( x.cols, x.rows );
  for ( let i = 0, l = x.cols; i < l; i++ ) {
    for ( let j = 0, m = x.rows; j < m; j++ ) {
      matrix.set ( i, j, x.get ( j, i ) );
    }
  }
  return matrix;
};

/* FUSED */

const fusedAddProductScale = ( x: Matrix, y: Matrix, z: Matrix, factor: number ): Matrix => {
  const matrix = new Matrix ( y.rows, z.cols );
  for ( let i = 0, l = y.rows; i < l; i++ ) {
    for ( let j = 0, m = z.cols; j < m; j++ ) {
      let sum = x.get ( i, j );
      for ( let k = 0, n = y.cols; k < n; k++ ) {
        sum += y.get ( i, k ) * z.get ( k, j ) * factor;
      }
      matrix.set ( i, j, sum );
    }
  }
  return matrix;
};

const fusedMultiplyMapActivation = ( x: Matrix, y: Matrix, activation: Identity<number> ): Matrix => {
  const matrix = new Matrix ( x.rows, x.cols );
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      matrix.set ( i, j, x.get ( i, j ) * activation ( y.get ( i, j ) ) );
    }
  }
  return matrix;
};

/* EXPORT */

export {abs, add, divide, each, each2, map, map2, mean, multiply, product, random, reduce, scale, subtract, sum, transpose};
export {fusedAddProductScale, fusedMultiplyMapActivation};
