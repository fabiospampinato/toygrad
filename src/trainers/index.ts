
/* IMPORT */

import Adadelta from '~/trainers/adadelta';
import Adagrad from '~/trainers/adagrad';
import Adam from '~/trainers/adam';
import Nesterov from '~/trainers/nesterov';
import SGD from '~/trainers/sgd';

/* MAIN */

const Trainers = {
  Adadelta,
  Adagrad,
  Adam,
  Nesterov,
  SGD
};

/* EXPORT */

export default Trainers;
