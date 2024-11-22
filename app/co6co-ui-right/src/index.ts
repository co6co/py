import installer from './defaults';
//export * from "..." 不会导出默认值
export * from './components';
export * from './constants';
export * from './directives';
export * from './utils';
export * from './hooks';
export * from './view';
export * from './api';

export * from './views';
export const install = installer.install;
export const version = installer.version;
export default installer;
