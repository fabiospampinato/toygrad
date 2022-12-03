
/* IMPORT */

import Matrix from './matrix';
import type {ActivationFN} from './types';

/* MAIN */

// These functions are pure, arguments are never mutated and the output depends solely on the input

const abs = ( x: Matrix ): Matrix => {
  return map ( x, Math.abs );
};

const activate = ( x: Matrix, activation: ActivationFN, derivative: boolean ): Matrix => {
  return map ( x, x => activation ( x, derivative ) );
};

const add = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x + y );
};

const ceil = ( x: Matrix ): Matrix => {
  return map ( x, Math.ceil );
};

const column = ( x: Matrix, index: number ): Matrix => {
  const matrix = new Matrix ( x.rows, 1 );
  return map ( matrix, ( _, row ) => x.get ( row, index ) );
};

const count = ( x: Matrix ): number => {
  return x.rows * x.cols;
};

const cube = ( x: Matrix ): Matrix => {
  return pow ( x, 3 );
};

const diagonal = ( x: Matrix ): Matrix => {
  const dimension = Math.min ( x.rows, x.cols );
  const matrix = new Matrix ( dimension, dimension );
  return map ( matrix, ( _, row, col ) => ( row === col ) ? x.get ( row, col ) : 0 );
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

const each2 = ( x: Matrix, y: Matrix, iterator: ( x: number, y: number, row: number, col: number ) => void ): void => {
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      iterator ( x.get ( i, j ), y.get ( i, j ), i, j );
    }
  }
};

const fill = ( x: Matrix, value: number ): Matrix => {
  return map ( x, () => value );
};

const floor = ( x: Matrix ): Matrix => {
  return map ( x, Math.floor );
};

const from = ( x: Matrix | number[][] ): Matrix => {
  if ( !Array.isArray ( x ) ) return x; // Fast-path, pre-allocated Matrix
  const rows = x.length;
  const cols = x[0].length;
  const matrix = new Matrix ( rows, cols );
  for ( let i = 0; i < rows; i++ ) {
    for ( let j = 0; j < cols; j++ ) {
      matrix.set ( i, j, x[i][j] );
    }
  }
  return matrix;
};

const identity = ( rows: number, cols: number ): Matrix => {
  const matrix = new Matrix ( rows, cols );
  return map ( matrix, ( _, row, col ) => ( row === col ) ? 1 : 0 );
};

const log2 = ( x: Matrix ): Matrix => {
  return map ( x, Math.log2 );
};

const log10 = ( x: Matrix ): Matrix => {
  return map ( x, Math.log10 );
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
  return sum ( x ) / count ( x );
};

const mae = ( x: Matrix, y: Matrix ): number => {
  return reduce2 ( x, y, ( acc, x, y ) => acc + Math.abs ( x - y ), 0 ) / count ( x );
};

const max = ( x: Matrix ): number => {
  return reduce ( x, ( acc, x ) => Math.max ( acc, x ), -Infinity );
};

const min = ( x: Matrix ): number => {
  return reduce ( x, ( acc, x ) => Math.min ( acc, x ), Infinity );
};

const modulo = ( x: Matrix, y: number ): Matrix => {
  return map ( x, x => x % y );
};

const mse = ( x: Matrix, y: Matrix ): number => {
  return reduce2 ( x, y, ( acc, x, y ) => acc + ( ( x - y ) ** 2 ), 0 ) / count ( x );
};

const multiply = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x * y );
};

const ones = ( rows: number, cols: number ): Matrix => {
  const matrix = new Matrix ( rows, cols );
  return fill ( matrix, 1 );
};

const pow = ( x: Matrix, exponent: number ): Matrix => {
  return map ( x, x => x ** exponent );
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

const reduce2 = <T> ( x: Matrix, y: Matrix, iterator: ( acc: T, x: number, y: number, row: number, col: number ) => T, acc: T ): T => {
  each2 ( x, y, ( x, y, row, col ) => {
    acc = iterator ( acc, x, y, row, col );
  });
  return acc;
};

const resize = ( x: Matrix, rows: number, cols: number ): Matrix => {
  const matrix = new Matrix ( rows, cols );
  return map ( matrix, ( _, row, col ) => ( row < x.rows && col < x.cols ) ? x.get ( row, col ) : 0 );
};

const round = ( x: Matrix ): Matrix => {
  return map ( x, Math.round );
};

const row = ( x: Matrix, index: number ): Matrix => {
  const matrix = new Matrix ( 1, x.cols );
  return map ( matrix, ( _, __, col ) => x.get ( index, col ) );
};

const scale = ( x: Matrix, factor: number ): Matrix => {
  if ( factor === 1 ) return x;
  return map ( x, x => x * factor );
};

const sign = ( x: Matrix ): Matrix => {
  return map ( x, Math.sign );
};

const size = ( x: Matrix ): [number, number] => {
  return [x.rows, x.cols];
};

const sqrt = ( x: Matrix ): Matrix => {
  return map ( x, Math.sqrt );
};

const square = ( x: Matrix ): Matrix => {
  return pow ( x, 2 );
};

const subtract = ( x: Matrix, y: Matrix ): Matrix => {
  return map2 ( x, y, ( x, y ) => x - y );
};

const sum = ( x: Matrix ): number => {
  return reduce ( x, ( acc, x ) => acc + x, 0 );
};

const trace = ( x: Matrix ): number => {
  return sum ( diagonal ( x ) );
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

const zeros = ( rows: number, cols: number ): Matrix => {
  return new Matrix ( rows, cols );
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

const fusedAddScale = ( x: Matrix, y: Matrix, factor: number ): Matrix => {
  const matrix = new Matrix ( x.rows, x.cols );
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = x.cols; j < m; j++ ) {
      matrix.set ( i, j, x.get ( i, j ) + ( y.get ( i, j ) * factor ) );
    }
  }
  return matrix;
};

const fusedProductBiased = ( x: Matrix, y: Matrix, biases: Matrix ): Matrix => {
  const matrix = new Matrix ( x.rows, y.cols );
  for ( let i = 0, l = x.rows; i < l; i++ ) {
    for ( let j = 0, m = y.cols; j < m; j++ ) {
      let sum = biases.get ( 0, j );
      for ( let k = 0, n = x.cols; k < n; k++ ) {
        sum += x.get ( i, k ) * y.get ( k, j );
      }
      matrix.set ( i, j, sum );
    }
  }
  return matrix;
};

/* EXPORT */

export {abs, activate, add, ceil, column, count, cube, diagonal, divide, each, each2, fill, floor, from, identity, log2, log10, map, map2, mean, mae, max, min, modulo, mse, multiply, ones, pow, product, random, reduce, reduce2, resize, round, row, scale, sign, size, sqrt, square, subtract, sum, trace, transpose, zeros};
export {fusedAddProductScale, fusedAddScale, fusedProductBiased};
