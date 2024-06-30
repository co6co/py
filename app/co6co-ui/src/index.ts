export { makeInstaller } from './make-installer';
import installer from './defaults';
export * from './components';
export * from './constants';
export * from './directives';
export * from './utils';
export * from './hooks';

export const install = installer.install;
export const version = installer.version;
export const PiniaInstance = installer.piniaInstance;
export default installer;
