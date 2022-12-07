
/* IMPORT */

import AbstractHidden from '~/layers/abstract_hidden';
import type {AbstractActivationOptions} from '~/types';

/* MAIN */

class AbstractActivation<T extends AbstractActivationOptions = AbstractActivationOptions> extends AbstractHidden<T> {}

/* EXPORT */

export default AbstractActivation;
