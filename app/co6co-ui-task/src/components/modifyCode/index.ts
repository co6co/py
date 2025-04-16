//import { withInstall } from 'co6co';
//import modifyCode from './src/modifyCode';
//export const ModifyCode = withInstall(modifyCode);

export * from './src';
import ModifyCode, { Item } from './src/modifyCode';
export default ModifyCode;
export type { Item };
