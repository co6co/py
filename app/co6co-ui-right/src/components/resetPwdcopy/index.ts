import { withInstall } from 'co6co';

import resetPwdcopy from './src/resetPwdcopy';

export const ResetPwdcopy = withInstall(resetPwdcopy);
export default ResetPwdcopy;

export * from './src';
