import installer from './defaults';
export * from './defaults';

//export * from "..." 不会导出默认值
export * from './components';
export * from './constants';
export * from './directives';
export * from './utils';
export * from './hooks';
export * from './view';
export * from './view/views';
export * from './api';
export const install = installer.install;
export default installer;
