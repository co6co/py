import { withInstall } from 'co6co';

import captcha from './src/captcha';

export const Captcha = withInstall(captcha);
export default Captcha;
export * from './src';
