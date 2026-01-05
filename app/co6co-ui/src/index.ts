// 恢复原始导出方式，但优化结构
export { Installer } from './make-installer';
import installer from './defaults';

// 导出组件、常量、指令、工具、钩子和视图
export * from './components';
export * from './constants';
export * from './directives';
export * from './utils';
export * from './hooks';
export * from './view';

// 导出installer的属性
export const install = installer.install;
export const version = installer.version;
export const piniaInstance = installer.piniaInstance;

// 默认导出
export default installer;
