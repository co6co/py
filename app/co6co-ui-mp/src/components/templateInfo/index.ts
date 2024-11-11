
import TemplateInfo from './src/templateInfo';
export type { Item as TemplateItem } from './src/templateInfo';
//只在当前模块中使用是否不用 Install
//import { withInstall } from 'co6co';
//export const TemplateInfo = withInstall(templateInfo);
export default TemplateInfo;

export * from './src';
