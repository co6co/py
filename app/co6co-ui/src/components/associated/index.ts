import { withInstall } from '../../utils';

import associated from './src/associated';
export const Associated = withInstall(associated);
export default Associated;

import transfer from './src/transfer';
export const Transfer = withInstall(transfer);

export * from './src';
