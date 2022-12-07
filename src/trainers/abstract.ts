
/* IMPORT */

import Buffer from '~/buffer';
import type NeuralNetwork from '~/neural_network';
import type Tensor from '~/tensor';
import type {TrainerAbstractOptions, TrainerResult} from '~/types';

/* MAIN */

//TODO: Actually split this up into multiple classes

class Abstract {

  /* VARIABLES */

  nn: NeuralNetwork;

  method: 'sgd' | 'adam' | 'adagrad' | 'adadelta' | 'nesterov';

  batchSize: number;
  learningRate: number;
  l1decay: number;
  l2decay: number;

  momentum: number;
  ro: number;
  eps: number;
  beta1: number;
  beta2: number;

  k: number;
  gsum: Float32Array[];
  xsum: Float32Array[];

  /* CONSTRUCTOR */

  constructor ( nn: NeuralNetwork, options: TrainerAbstractOptions = {} ) {

    this.nn = nn;

    this.method = options.method ?? 'sgd';

    this.batchSize = options.batchSize ?? 1;
    this.learningRate = options.learningRate ?? 0.01;
    this.l1decay = options.l1decay ?? 0;
    this.l2decay = options.l2decay ?? 0;

    this.momentum = options.momentum ?? 0.9;
    this.ro = options.ro ?? 0.95;
    this.eps = options.eps ?? 1e-8;
    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;

    this.k = 0;
    this.gsum = [];
    this.xsum = [];

  }

  /* API */

  train ( input: Tensor, output: ArrayLike<number> | number ): TrainerResult {

    this.nn.forward ( input, true );

    let cost = this.nn.backward(output as any); //TSC
    let l1loss = 0;
    let l2loss = 0;

    this.k++;
    if(this.k % this.batchSize === 0) {

      let pglist = this.nn.getParamsAndGrads();

      // initialize lists for accumulators. Will only be done once on first iteration
      if(this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0)) {
        // only vanilla sgd doesnt need either lists
        // momentum needs gsum
        // adagrad needs gsum
        // adam and adadelta needs gsum and xsum
        for(let i=0;i<pglist.length;i++) {
          this.gsum.push(new Buffer(pglist[i].params.length));
          if(this.method === 'adam' || this.method === 'adadelta') {
            this.xsum.push(new Buffer(pglist[i].params.length));
          } else {
            // this.xsum.push([]); // conserve memory
          }
        }
      }

      // perform an update for all sets of weights
      for ( let i = 0; i < pglist.length; i++ ) {
        let pg = pglist[i]; // param, gradient, other options in future (custom learning rate etc)
        let p = pg.params;
        let g = pg.grads;

        // learning rate for some parameters.
        let l2decay = pg.l2decay ?? 1;
        let l1decay = pg.l1decay ?? 1;
        let l2_decay = this.l2decay * l2decay;
        let l1_decay = this.l1decay * l1decay;

        let plen = p.length;
        for(let j=0;j<plen;j++) {
          l2loss += l2_decay*p[j]*p[j]/2; // accumulate weight decay loss
          l1loss += l1_decay*Math.abs(p[j]);
          let l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
          let l2grad = l2_decay * (p[j]);

          let gij = (l2grad + l1grad + g[j]) / this.batchSize; // raw batch gradient

          let gsumi = this.gsum[i];
          let xsumi = this.xsum[i];
          if(this.method === 'adam') {
            gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
            xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
            let biasCorr1 = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
            let biasCorr2 = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
            let dx =  - this.learningRate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
            p[j] += dx;
          } else if(this.method === 'adagrad') {
            gsumi[j] = gsumi[j] + gij * gij;
            let dx = - this.learningRate / Math.sqrt(gsumi[j] + this.eps) * gij;
            p[j] += dx;
          } else if(this.method === 'adadelta') {
            gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
            let dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
            xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
            p[j] += dx;
          } else if(this.method === 'nesterov') {
            let dx = gsumi[j];
            gsumi[j] = gsumi[j] * this.momentum + this.learningRate * gij;
              dx = this.momentum * dx - (1 + this.momentum) * gsumi[j];
              p[j] += dx;
          } else if (this.method === 'sgd') {
            if(this.momentum > 0) {
              // momentum update
              let dx = this.momentum * gsumi[j] - this.learningRate * gij; // step
              gsumi[j] = dx; // back this up for next iteration of momentum
              p[j] += dx; // apply corrected gradient
            } else {
              // vanilla sgd
              p[j] +=  - this.learningRate * gij;
            }
          } else {
            throw new Error ( 'Unsupposed trainer' );
          }
          g[j] = 0; // zero out gradient so that we can begin accumulating anew
        }
      }
    }

    const loss = cost + l1loss + l2loss;
    // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
    // in future, TODO: have to completely redo the way loss is done around the network as currently
    // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
    // and it should all be computed correctly and automatically.
    return {l2loss, l1loss, cost, loss};

  }

}

/* EXPORT */

export default Abstract;
