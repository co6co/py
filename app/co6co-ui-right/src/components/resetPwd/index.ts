import { withInstall } from 'co6co';

import resetPwd from './src/resetPwd';

export const ResetPwd = withInstall(resetPwd);
export default ResetPwd;

export * from './src';
