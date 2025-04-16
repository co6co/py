//import { withInstall } from 'co6co';
//import modifyTask from './src/modifyTask';
//export const ModifyTask = withInstall(modifyTask);
export * from './src';
import ModifyTask, { Item } from './src/modifyTask';
export default ModifyTask;
export type { Item };
